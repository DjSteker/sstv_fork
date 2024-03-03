
"""Clase y métodos para decodificar la señal SSTV."""

import numpy as np
import soundfile
from PIL import Image
from scipy.signal.windows import hann

from . import spec
from .common import log_message, progress_bar


def calc_lum(freq):
    """Convierte el rango de frecuencia de píxeles SSTV en 0-255 bytes de luminancia"""

    lum = int(round((freq - 1500) / 3.1372549))
    return min(max(lum, 0), 255)


def barycentric_peak_interp(bins, x):
    """Interpolar entre contenedores de frecuencia para encontrar el valor x del pico"""

    # Toma x como índice del contenedor más grande e interpola el
    # x valor del pico usando vecinos en la matriz de contenedores

    # Asegúrese de que los datos estén dentro de los límites
    y1 = bins[x] if x <= 0 else bins[x-1]
    y3 = bins[x] if x + 1 >= len(bins) else bins[x+1]

    denom = y3 + bins[x] + y1
    if denom == 0:
        return 0  # erroneous

    return (y3 - y1) / denom + x


class SSTVDecoder(object):

    """Cree un decodificador SSTV para decodificar datos de audio"""

    def __init__(self, audio_file):
        self.mode = None

        self._audio_file = audio_file

        self._samples, self._sample_rate = soundfile.read(self._audio_file)

        if self._samples.ndim > 1:  # convert to mono if stereo
            self._samples = self._samples.mean(axis=1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self.close()

    def __del__(self):
        self.close()

    def decode(self, skip=0.0):
        """Intenta decodificar los datos de audio como una señal SSTV
         Devuelve una imagen PIL en caso de éxito y Ninguna si no se encontró ninguna señal SSTV
         """

        if skip > 0.0:
            self._samples = self._samples[round(skip * self._sample_rate):]

        header_end = self._find_header()

        if header_end is None:
            return None

        self.mode = self._decode_vis(header_end)

        vis_end = header_end + round(spec.VIS_BIT_SIZE * 9 * self._sample_rate)

        image_data = self._decode_image_data(vis_end)

        return self._draw_image(image_data)

    def close(self):
        """Cierra cualquier archivo de entrada si existe"""

        if self._audio_file is not None and not self._audio_file.closed:
            self._audio_file.close()

    def _peak_fft_freq(self, data):
        """Encuentra la frecuencia máxima de una sección de datos de audio"""

        windowed_data = data * hann(len(data))
        fft = np.abs(np.fft.rfft(windowed_data))

        # Obtener el índice del contenedor de mayor magnitu
        x = np.argmax(fft)
        # Frecuencia máxima interpolada
        peak = barycentric_peak_interp(fft, x)

        # Return frequency in hz
        return peak * self._sample_rate / len(windowed_data)

    def _find_header(self):
        """Encuentra la muestra aproximada del final del encabezado de calibración"""

        header_size = round(spec.HDR_SIZE * self._sample_rate)
        window_size = round(spec.HDR_WINDOW_SIZE * self._sample_rate)

        # Desplazamientos de muestra relativos de los tonos del encabezado
        leader_1_sample = 0
        leader_1_search = leader_1_sample + window_size

        break_sample = round(spec.BREAK_OFFSET * self._sample_rate)
        break_search = break_sample + window_size

        leader_2_sample = round(spec.LEADER_OFFSET * self._sample_rate)
        leader_2_search = leader_2_sample + window_size

        vis_start_sample = round(spec.VIS_START_OFFSET * self._sample_rate)
        vis_start_search = vis_start_sample + window_size

        jump_size = round(0.002 * self._sample_rate)  # check every 2ms

        # El margen de error creado aquí será insignificante al decodificar el
        # vis debido a que cada bit tiene una longitud de 30 ms. Arreglamos este margen de error
        # al decodificar la imagen alineando cada pulso de sincronización

        for current_sample in range(0, len(self._samples) - header_size,
                                    jump_size):
            # Actualizar mensaje de progreso de búsqueda
            if current_sample % (jump_size * 256) == 0:
                search_msg = "Searching for calibration header... {:.1f}s"
                progress = current_sample / self._sample_rate
                log_message(search_msg.format(progress), recur=True)

            search_end = current_sample + header_size
            search_area = self._samples[current_sample:search_end]

            leader_1_area = search_area[leader_1_sample:leader_1_search]
            break_area = search_area[break_sample:break_search]
            leader_2_area = search_area[leader_2_sample:leader_2_search]
            vis_start_area = search_area[vis_start_sample:vis_start_search]

            # Comprueba que sean las frecuencias correctas.
            if (abs(self._peak_fft_freq(leader_1_area) - 1900) < 50
               and abs(self._peak_fft_freq(break_area) - 1200) < 50
               and abs(self._peak_fft_freq(leader_2_area) - 1900) < 50
               and abs(self._peak_fft_freq(vis_start_area) - 1200) < 50):

                stop_msg = "Searching for calibration header... Found!{:>4}"
                log_message(stop_msg.format(' '))
                return current_sample + header_size

        log_message()
        log_message("No se pudo encontrar el encabezado SSTV en el archivo de audio proporcionado",
                    err=True)
        return None

    def _decode_vis(self, vis_start):
        """Decodifica el vis de los datos de audio y devuelve el modo SSTV"""

        bit_size = round(spec.VIS_BIT_SIZE * self._sample_rate)
        vis_bits = []

        for bit_idx in range(8):
            bit_offset = vis_start + bit_idx * bit_size
            section = self._samples[bit_offset:bit_offset+bit_size]
            freq = self._peak_fft_freq(section)
            # 1100 hz = 1, 1300hz = 0
            vis_bits.append(int(freq <= 1200))

        # Compruebe si hay paridad uniforme en el último bit
        parity = sum(vis_bits) % 2 == 0
        if not parity:
            raise ValueError("Error decoding VIS header (invalid parity bit)")

        # LSB primero por lo que debemos invertir e ignorar el bit de paridad
        vis_value = 0
        for bit in vis_bits[-2::-1]:
            vis_value = (vis_value << 1) | bit

        if vis_value not in spec.VIS_MAP:
            error = "SSTV mode is unsupported (VIS: {})"
            raise ValueError(error.format(vis_value))

        mode = spec.VIS_MAP[vis_value]
        log_message("Modo SSTV detectado {}".format(mode.NAME))

        return mode

    def _align_sync(self, align_start, start_of_sync=True):
        """Devuelve una muestra donde se encontró el comienzo del pulso de sincronización"""

        # TODO - improve this

        sync_window = round(self.mode.SYNC_PULSE * 1.4 * self._sample_rate)
        align_stop = len(self._samples) - sync_window

        if align_stop <= align_start:
            return None  # Reached end of audio

        for current_sample in range(align_start, align_stop):
            section_end = current_sample + sync_window
            search_section = self._samples[current_sample:section_end]

            if self._peak_fft_freq(search_section) > 1350:
                break

        end_sync = current_sample + (sync_window // 2)

        if start_of_sync:
            return end_sync - round(self.mode.SYNC_PULSE * self._sample_rate)
        else:
            return end_sync

    def _decode_image_data(self, image_start):
        """Decodifica la imagen de la sección de transmisión de una señal sstv"""

        window_factor = self.mode.WINDOW_FACTOR
        centre_window_time = (self.mode.PIXEL_TIME * window_factor) / 2
        pixel_window = round(centre_window_time * 2 * self._sample_rate)

        height = self.mode.LINE_COUNT
        channels = self.mode.CHAN_COUNT
        width = self.mode.LINE_WIDTH
        # Utilice la comprensión de listas para iniciar la lista para que podamos devolver los datos antes
        image_data = [[[0 for i in range(width)]
                       for j in range(channels)] for k in range(height)]

        seq_start = image_start
        if self.mode.HAS_START_SYNC:
            # Comience al final del pulso de sincronización inicial
            seq_start = self._align_sync(image_start, start_of_sync=False)
            if seq_start is None:
                raise EOFError("Se alcanzó el final del audio antes que los datos de la imagen")

        for line in range(height):

            if self.mode.CHAN_SYNC > 0 and line == 0:
                # Alinear seq_start con el comienzo del pulso de sincronización anterior
                sync_offset = self.mode.CHAN_OFFSETS[self.mode.CHAN_SYNC]
                seq_start -= round((sync_offset + self.mode.SCAN_TIME)
                                   * self._sample_rate)

            for chan in range(channels):

                if chan == self.mode.CHAN_SYNC:
                    if line > 0 or chan > 0:
                        # Establecer desplazamiento base a la siguiente línea
                        seq_start += round(self.mode.LINE_TIME *
                                           self._sample_rate)

                    # Alinear con el inicio del pulso de sincronización
                    seq_start = self._align_sync(seq_start)
                    if seq_start is None:
                        log_message()
                        log_message("Se alcanzó el final del audio durante la decodificación.")
                        return image_data

                pixel_time = self.mode.PIXEL_TIME
                if self.mode.HAS_HALF_SCAN:
                    # El modo robot tiene un segundo/tercer escaneo de longitud media
                    if chan > 0:
                        pixel_time = self.mode.HALF_PIXEL_TIME

                    centre_window_time = (pixel_time * window_factor) / 2
                    pixel_window = round(centre_window_time * 2 *
                                         self._sample_rate)

                for px in range(width):

                    chan_offset = self.mode.CHAN_OFFSETS[chan]

                    px_pos = round(seq_start + (chan_offset + px *
                                   pixel_time - centre_window_time) *
                                   self._sample_rate)
                    px_end = px_pos + pixel_window

                    # Si estamos ejecutando fft más allá de la duración del audio, deténgase antes
                    if px_end >= len(self._samples):
                        log_message()
                        log_message("Se alcanzó el final del audio durante la decodificación.")
                        return image_data

                    pixel_area = self._samples[px_pos:px_end]
                    freq = self._peak_fft_freq(pixel_area)

                    image_data[line][chan][px] = calc_lum(freq)

            progress_bar(line, height - 1, "Decodificando imagen...")

        return image_data

    def _draw_image(self, image_data):
        """Representa la imagen a partir de la señal sstv decodificada"""

        # Let PIL do YUV-RGB conversion for us
        if self.mode.COLOR == spec.COL_FMT.YUV:
            col_mode = "YCbCr"
        else:
            col_mode = "RGB"

        width = self.mode.LINE_WIDTH
        height = self.mode.LINE_COUNT
        channels = self.mode.CHAN_COUNT

        image = Image.new(col_mode, (width, height))
        pixel_data = image.load()

        log_message("Drawing image data...")

        for y in range(height):

            odd_line = y % 2
            for x in range(width):

                if channels == 2:

                    if self.mode.HAS_ALT_SCAN:
                        if self.mode.COLOR == spec.COL_FMT.YUV:
                            # R36
                            pixel = (image_data[y][0][x],
                                     image_data[y-(odd_line-1)][1][x],
                                     image_data[y-odd_line][1][x])

                elif channels == 3:

                    if self.mode.COLOR == spec.COL_FMT.GBR:
                        # M1, M2, S1, S2, SDX
                        pixel = (image_data[y][2][x],
                                 image_data[y][0][x],
                                 image_data[y][1][x])
                    elif self.mode.COLOR == spec.COL_FMT.YUV:
                        # R72
                        pixel = (image_data[y][0][x],
                                 image_data[y][2][x],
                                 image_data[y][1][x])
                    elif self.mode.COLOR == spec.COL_FMT.RGB:
                        pixel = (image_data[y][0][x],
                                 image_data[y][1][x],
                                 image_data[y][2][x])

                pixel_data[x, y] = pixel

        if image.mode != "RGB":
            image = image.convert("RGB")

        log_message("...¡Hecho!")
        return image
