/* Core/Src/bridge_manager.c */

#include "bridge_manager.h"
#include "usart.h"
#include "Mission_Data.h"
#include "GPS_Manager.h"
#include "Sensors_Manager.h"
#include "Mission_Manager.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

extern UART_HandleTypeDef huart2;
extern Mission_Data_t cansatData;

/* Dirección del LoRa del PC */
#define LORA_DEST_ADDRESS   2

/* El RYLR998 permite máximo 240 bytes de payload */
#define LORA_MAX_PAYLOAD    240

/*
 * Estructura simple para guardar datos GPS limpios.
 * Ahora todos los valores inválidos se mandan como "0",
 * no como "NA", para facilitar la interfaz gráfica.
 */
typedef struct
{
    char fix[4];        /* 0 = sin fix, 1 = fix válido */
    char time[16];      /* hora UTC o 0 */
    char lat[20];       /* latitud decimal o 0 */
    char lon[20];       /* longitud decimal o 0 */
    char sat[6];        /* satélites o 0 */
    char hdop[12];      /* precisión horizontal o 0 */
    char gps_alt[16];   /* altitud GPS o 0 */
    char speed[16];     /* velocidad km/h o 0 */
    char course[16];    /* rumbo o 0 */
} GPS_Clean_t;

/*
 * Valores iniciales.
 * Antes tenías "NA"; ahora todo inicia en "0".
 */
static GPS_Clean_t gpsData = {
    "0", "0", "0", "0", "0", "0", "0", "0", "0"
};

/*
 * Elimina \r y \n de una cadena.
 */
static void Remove_CR_LF(char *str)
{
    int i;

    for (i = 0; str[i] != '\0'; i++)
    {
        if (str[i] == '\r' || str[i] == '\n')
        {
            str[i] = '\0';
            break;
        }
    }
}

/*
 * Obtiene un campo NMEA sin perder campos vacíos.
 */
static void NMEA_GetField(const char *sentence, int field_index, char *out, int out_size)
{
    int current_field = 0;
    int j = 0;
    const char *p = sentence;

    if (out_size <= 0)
    {
        return;
    }

    out[0] = '\0';

    while (*p != '\0' && current_field < field_index)
    {
        if (*p == ',')
        {
            current_field++;
        }
        p++;
    }

    if (current_field != field_index)
    {
        return;
    }

    while (*p != '\0' && *p != ',' && *p != '*' && j < (out_size - 1))
    {
        out[j] = *p;
        j++;
        p++;
    }

    out[j] = '\0';
}

/*
 * Verifica si una sentencia es GGA, GLL, VTG, etc.
 * Sirve para $GPGGA, $GNGGA, $GAGGA, etc.
 */
static int NMEA_IsType(const char *sentence, const char *type)
{
    if (sentence == NULL)
    {
        return 0;
    }

    if (strlen(sentence) < 6)
    {
        return 0;
    }

    if (sentence[0] != '$')
    {
        return 0;
    }

    if (sentence[3] == type[0] &&
        sentence[4] == type[1] &&
        sentence[5] == type[2])
    {
        return 1;
    }

    return 0;
}

/*
 * Si un campo quedó vacío, lo cambia a "0".
 */
static void Set_Zero_If_Empty(char *str, int size)
{
    if (str == NULL || size <= 0)
    {
        return;
    }

    if (str[0] == '\0')
    {
        snprintf(str, size, "0");
    }
}

/*
 * Fuerza todos los datos GPS que dependen del fix a cero.
 * Esto se usa cuando el GPS no tiene posición válida.
 */
static void GPS_SetInvalidDataToZero(void)
{
    snprintf(gpsData.fix, sizeof(gpsData.fix), "0");
    snprintf(gpsData.lat, sizeof(gpsData.lat), "0");
    snprintf(gpsData.lon, sizeof(gpsData.lon), "0");
    snprintf(gpsData.sat, sizeof(gpsData.sat), "0");
    snprintf(gpsData.hdop, sizeof(gpsData.hdop), "0");
    snprintf(gpsData.gps_alt, sizeof(gpsData.gps_alt), "0");
    snprintf(gpsData.speed, sizeof(gpsData.speed), "0");
    snprintf(gpsData.course, sizeof(gpsData.course), "0");

    /*
     * La hora GPS puede existir aunque no haya fix.
     * Por eso no la borramos aquí.
     * Pero si está vacía, la dejamos en 0.
     */
    Set_Zero_If_Empty(gpsData.time, sizeof(gpsData.time));
}

/*
 * Convierte coordenadas NMEA a grados decimales.
 *
 * NMEA:
 * Latitud:  ddmm.mmmm
 * Longitud: dddmm.mmmm
 *
 * Ejemplo:
 * 0330.1234,N  ->  3.502056
 * 07630.5678,W -> -76.509463
 */
static void NMEA_ToDecimal(const char *value, const char *hemisphere, char *out, int out_size)
{
    double raw;
    int degrees;
    double minutes;
    double decimal;

    if (value == NULL || hemisphere == NULL || out == NULL)
    {
        return;
    }

    if (value[0] == '\0' || hemisphere[0] == '\0')
    {
        snprintf(out, out_size, "0");
        return;
    }

    raw = atof(value);

    if (raw <= 0.0)
    {
        snprintf(out, out_size, "0");
        return;
    }

    degrees = (int)(raw / 100.0);
    minutes = raw - ((double)degrees * 100.0);
    decimal = (double)degrees + (minutes / 60.0);

    if (hemisphere[0] == 'S' || hemisphere[0] == 'W')
    {
        decimal = -decimal;
    }

    snprintf(out, out_size, "%.6f", decimal);
}

/*
 * Actualiza los datos GPS limpios según la sentencia NMEA actual.
 * No manda la NMEA completa, solo extrae lo importante.
 */
static void GPS_UpdateCleanData(const char *nmea_sentence)
{
    char field[24];

    char lat_raw[20];
    char lat_hemi[4];
    char lon_raw[20];
    char lon_hemi[4];

    char fix_quality[8];
    char status[8];

    if (nmea_sentence == NULL)
    {
        GPS_SetInvalidDataToZero();
        return;
    }

    /*
     * GGA:
     * $GPGGA,time,lat,N/S,lon,E/W,fix,sat,hdop,alt,...
     */
    if (NMEA_IsType(nmea_sentence, "GGA"))
    {
        NMEA_GetField(nmea_sentence, 1, gpsData.time, sizeof(gpsData.time));

        NMEA_GetField(nmea_sentence, 2, lat_raw, sizeof(lat_raw));
        NMEA_GetField(nmea_sentence, 3, lat_hemi, sizeof(lat_hemi));
        NMEA_GetField(nmea_sentence, 4, lon_raw, sizeof(lon_raw));
        NMEA_GetField(nmea_sentence, 5, lon_hemi, sizeof(lon_hemi));

        NMEA_GetField(nmea_sentence, 6, fix_quality, sizeof(fix_quality));
        NMEA_GetField(nmea_sentence, 7, gpsData.sat, sizeof(gpsData.sat));
        NMEA_GetField(nmea_sentence, 8, gpsData.hdop, sizeof(gpsData.hdop));
        NMEA_GetField(nmea_sentence, 9, gpsData.gps_alt, sizeof(gpsData.gps_alt));

        Set_Zero_If_Empty(gpsData.time, sizeof(gpsData.time));
        Set_Zero_If_Empty(gpsData.sat, sizeof(gpsData.sat));
        Set_Zero_If_Empty(gpsData.hdop, sizeof(gpsData.hdop));
        Set_Zero_If_Empty(gpsData.gps_alt, sizeof(gpsData.gps_alt));

        /*
         * fix_quality:
         * 0 = inválido
         * 1 = GPS fix
         * 2 = DGPS fix
         */
        if (fix_quality[0] == '0' || fix_quality[0] == '\0')
        {
            GPS_SetInvalidDataToZero();
        }
        else
        {
            snprintf(gpsData.fix, sizeof(gpsData.fix), "1");

            NMEA_ToDecimal(lat_raw, lat_hemi, gpsData.lat, sizeof(gpsData.lat));
            NMEA_ToDecimal(lon_raw, lon_hemi, gpsData.lon, sizeof(gpsData.lon));

            Set_Zero_If_Empty(gpsData.lat, sizeof(gpsData.lat));
            Set_Zero_If_Empty(gpsData.lon, sizeof(gpsData.lon));
            Set_Zero_If_Empty(gpsData.gps_alt, sizeof(gpsData.gps_alt));
        }
    }

    /*
     * GLL:
     * $GPGLL,lat,N/S,lon,E/W,time,status,mode*checksum
     *
     * status:
     * A = válido
     * V = inválido
     */
    else if (NMEA_IsType(nmea_sentence, "GLL"))
    {
        NMEA_GetField(nmea_sentence, 1, lat_raw, sizeof(lat_raw));
        NMEA_GetField(nmea_sentence, 2, lat_hemi, sizeof(lat_hemi));
        NMEA_GetField(nmea_sentence, 3, lon_raw, sizeof(lon_raw));
        NMEA_GetField(nmea_sentence, 4, lon_hemi, sizeof(lon_hemi));
        NMEA_GetField(nmea_sentence, 5, gpsData.time, sizeof(gpsData.time));
        NMEA_GetField(nmea_sentence, 6, status, sizeof(status));

        Set_Zero_If_Empty(gpsData.time, sizeof(gpsData.time));

        if (status[0] == 'A')
        {
            snprintf(gpsData.fix, sizeof(gpsData.fix), "1");

            NMEA_ToDecimal(lat_raw, lat_hemi, gpsData.lat, sizeof(gpsData.lat));
            NMEA_ToDecimal(lon_raw, lon_hemi, gpsData.lon, sizeof(gpsData.lon));

            Set_Zero_If_Empty(gpsData.lat, sizeof(gpsData.lat));
            Set_Zero_If_Empty(gpsData.lon, sizeof(gpsData.lon));
        }
        else
        {
            GPS_SetInvalidDataToZero();
        }
    }

    /*
     * VTG:
     * $GPVTG,course,T,,M,speed_knots,N,speed_kmh,K,mode*checksum
     */
    else if (NMEA_IsType(nmea_sentence, "VTG"))
    {
        NMEA_GetField(nmea_sentence, 1, gpsData.course, sizeof(gpsData.course));
        NMEA_GetField(nmea_sentence, 7, gpsData.speed, sizeof(gpsData.speed));

        Set_Zero_If_Empty(gpsData.course, sizeof(gpsData.course));
        Set_Zero_If_Empty(gpsData.speed, sizeof(gpsData.speed));
    }

    /*
     * Si llega otra sentencia, no hacemos nada.
     * Pero se deja el campo identificado por si después quieres depurar.
     */
    else
    {
        NMEA_GetField(nmea_sentence, 0, field, sizeof(field));
    }
}

void Bridge_Update(void)
{
    char payload[LORA_MAX_PAYLOAD + 1];
    char command[320];

    char gps_clean[120];

    int payload_len;
    int command_len;

    /*
     * Copiamos la sentencia GPS actual y le quitamos \r\n.
     */
    strncpy(gps_clean, cansatData.gps_sentence, sizeof(gps_clean) - 1);
    gps_clean[sizeof(gps_clean) - 1] = '\0';
    Remove_CR_LF(gps_clean);

    /*
     * Actualizamos datos GPS limpios.
     */
    GPS_UpdateCleanData(gps_clean);

    /*
     * Trama limpia para la estación terrena.
     *
     * Ahora todos los campos GPS inválidos salen como 0.
     */
    payload_len = snprintf(payload, sizeof(payload),
        "CS|ST=%d|AX=%.2f|AY=%.2f|AZ=%.2f|GX=%.2f|GY=%.2f|GZ=%.2f|"
        "T=%.2f|P=%.2f|ALT=%.2f|GF=%s|GT=%s|LAT=%s|LON=%s|SAT=%s|HDOP=%s|GALT=%s|SPD=%s|CRS=%s",
        currentState,
        cansatData.ax,
        cansatData.ay,
        cansatData.az,
        cansatData.gyro_x,
        cansatData.gyro_y,
        cansatData.gyro_z,
        cansatData.temperature,
        cansatData.pressure,
        cansatData.altitude,
        gpsData.fix,
        gpsData.time,
        gpsData.lat,
        gpsData.lon,
        gpsData.sat,
        gpsData.hdop,
        gpsData.gps_alt,
        gpsData.speed,
        gpsData.course
    );

    if (payload_len < 0)
    {
        return;
    }

    /*
     * Si se pasa de 240 caracteres, snprintf recorta.
     * Usamos strlen para obtener la longitud real enviada.
     */
    if (payload_len >= (int)sizeof(payload))
    {
        payload_len = strlen(payload);
    }

    /*
     * Comando AT para el RYLR998.
     */
    command_len = snprintf(command, sizeof(command),
        "AT+SEND=%d,%d,%s\r\n",
        LORA_DEST_ADDRESS,
        payload_len,
        payload
    );

    if (command_len < 0 || command_len >= (int)sizeof(command))
    {
        return;
    }

    HAL_UART_Transmit(&huart2, (uint8_t*)command, command_len, 1000);
}
