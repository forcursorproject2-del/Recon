import pdfplumber
import fitz  # PyMuPDF
import logging
from typing import Dict, Any, List, Optional
import os
from pathlib import Path
import glob
import hashlib
import pickle
import time
from .ocr_cache import OcrCache
import re
from .paddle_ocr import PaddleOCRExtractor
from .image_preprocessor import ImagePreprocessor
from .preprocessors.adaptive_engine import AdaptivePreprocessingEngine
from .preprocessors.light_preprocessor import LightPreprocessor
from .preprocessors.heavy_preprocessor import HeavyPreprocessor
from .preprocessors.photo_preprocessor import PhotoPreprocessor
from .preprocessors.universal_preprocessor import UniversalPreprocessor
from .ocr_quality_estimator import OCRQualityEstimator, OCRResultCombiner
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Поддерживаемые форматы изображений
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']

class PDFBatchProcessor:
    def __init__(self, use_ocr: bool = False, config_manager=None):
        self.use_ocr = use_ocr
        self.extractor = PDFTextExtractor(use_ocr=use_ocr, config_manager=config_manager)

    def process_folder(self, folder_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Processes all PDF and image files in a folder.
        Returns a dictionary mapping filenames to their extracted data.
        """
        logger.debug(f"Starting batch processing of folder: {folder_path}")
        
        # Находим PDF файлы
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        
        # Находим изображения
        image_files = []
        for ext in SUPPORTED_IMAGE_FORMATS:
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
        
        total_files = len(pdf_files) + len(image_files)
        logger.info(f"Found {len(pdf_files)} PDF files and {len(image_files)} image files to process")
        
        results = {}
        
        # Обрабатываем PDF файлы
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            try:
                logger.debug(f"Processing PDF file: {filename}")
                extracted_data = self.extractor.extract_text_with_structure(pdf_path)
                results[filename] = extracted_data
                logger.debug(f"Successfully processed: {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                results[filename] = {'error': str(e)}
        
        # Обрабатываем изображения
        for image_path in image_files:
            filename = os.path.basename(image_path)
            try:
                logger.debug(f"Processing image file: {filename}")
                extracted_data = self.extractor.extract_text_from_image(image_path)
                results[filename] = extracted_data
                logger.debug(f"Successfully processed: {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                results[filename] = {'error': str(e)}
        
        logger.info(f"Batch processing completed. Processed {len(results)} files.")
        return results

class PDFTextExtractor:
    def __init__(self, config_manager=None, use_ocr=False):
        self.config_manager = config_manager
        self.use_ocr = use_ocr

        self.ocr_cache = OcrCache()
        self.ocr_extractor = PaddleOCRExtractor(cache_db="ocr_cache.db")
        
        # Получаем настройки предобработки из конфига
        preprocess_config = {}
        if config_manager:
            preprocess_config = config_manager.config.get('image_preprocessing', {})
        
        self.image_preprocessor = ImagePreprocessor(preprocess_config)
        
        # Получаем полный конфиг для передачи в препроцессоры
        full_config = config_manager.config if config_manager else {}
        
        # Инициализация новой системы предобработки
        self.adaptive_engine = AdaptivePreprocessingEngine(config=full_config)
        self.light_preprocessor = LightPreprocessor(self.image_preprocessor)
        self.heavy_preprocessor = HeavyPreprocessor(config=full_config)
        
        # Инициализация препроцессора для фото
        photo_config = {}
        if config_manager:
            photo_config = config_manager.config.get('photo_preprocessing', {})
        self.photo_preprocessor = PhotoPreprocessor(photo_config)
        
        # Инициализация универсального препроцессора
        self.universal_preprocessor = UniversalPreprocessor(config=full_config)
        
        # Инициализация оценщика качества и комбинера результатов
        self.quality_estimator = OCRQualityEstimator()
        self.result_combiner = OCRResultCombiner()
        
        # Настройки из конфига
        self.use_confidence_selection = False
        self.heavy_preprocessing_enabled = True
        if config_manager:
            ocr_config = config_manager.config.get('ocr', {})
            self.use_confidence_selection = ocr_config.get('use_confidence_selection', False)
            self.heavy_preprocessing_enabled = ocr_config.get('heavy_preprocessing', True)
        
        # Кэш предобработанных изображений
        self.preprocessing_cache = {}
        self.cache_file = "preprocessing_cache.pkl"
        self._load_preprocessing_cache()

    def _get_image_hash(self, image: np.ndarray) -> str:
        """Вычисляет MD5 хэш изображения."""
        _, buffer = cv2.imencode('.png', image)
        return hashlib.md5(buffer.tobytes()).hexdigest()
    
    def _load_preprocessing_cache(self):
        """Загружает кэш из файла."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.preprocessing_cache = pickle.load(f)
                logger.debug(f"Загружен кэш предобработки: {len(self.preprocessing_cache)} записей")
            except Exception as e:
                logger.warning(f"Ошибка загрузки кэша предобработки: {e}")
                self.preprocessing_cache = {}
    
    def _save_preprocessing_cache(self):
        """Сохраняет кэш в файл."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.preprocessing_cache, f)
            logger.debug(f"Сохранен кэш предобработки: {len(self.preprocessing_cache)} записей")
        except Exception as e:
            logger.warning(f"Ошибка сохранения кэша предобработки: {e}")
    
    def extract_text_with_structure(self, pdf_path: str) -> Dict[str, object]:
        """
        Extracts text and structure from PDF. Uses OCR if text is short.
        Returns dict with keys: pages (list), metadata (dict), full_text (str).
        """
        from core_parser.utils.validators import validate_file_path, safe_file_size, ValidationError, SecurityError
        
        logger.debug(f"Starting extraction for {pdf_path}, use_ocr: True")
        
        # Валидация пути
        try:
            validated_path = validate_file_path(pdf_path, allowed_extensions=['.pdf'])
        except (ValidationError, SecurityError) as e:
            logger.error(f"Ошибка валидации пути: {e}")
            raise ValueError(f"Некорректный путь к файлу: {pdf_path}") from e
        
        # Проверка размера файла
        is_valid, size_mb = safe_file_size(validated_path, max_size_mb=10.0)
        if not is_valid:
            raise ValueError(f"FILE_TOO_LARGE: {pdf_path} ({size_mb:.1f} MB, максимум 10 MB)")

        try:
            with pdfplumber.open(str(validated_path)) as pdf:
                pages = []
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    page_data = {
                        'text': text,
                        'words': page.extract_words(),
                        'bbox': page.bbox,
                        'tables': page.extract_tables()
                    }
                    pages.append(page_data)
                    full_text += text + "\n"
                metadata = pdf.metadata
                # Добавляем имя файла в метаданные для классификации
                if metadata is None:
                    metadata = {}
                metadata['filename'] = Path(pdf_path).name
                metadata['file_path'] = str(pdf_path)
        except Exception as e:
            logger.warning(f"pdfplumber failed for {validated_path}: {e}. Using PyMuPDF fallback.")
            return self._extract_with_fitz(str(validated_path))

        MIN_ACCEPTABLE_LENGTH = 300  # threshold length to trigger OCR
        if len(full_text.strip()) < MIN_ACCEPTABLE_LENGTH:
            logger.info(f"Extracted text too short ({len(full_text)}) characters, using OCR.")
            full_text = self._extract_with_ocr(pdf_path, initial_text=full_text)
            # OCR returns unstructured text
            pages = [{'text': full_text, 'words': [], 'bbox': None, 'tables': []}]

        logger.debug(f"Full extracted text length: {len(full_text)}")
        logger.debug(f"Excerpt: {full_text[:1000]}")
        result = {
            'pages': pages,
            'metadata': metadata or {},
            'full_text': full_text
        }
        # Добавляем имя файла в структуру для классификации
        result['filename'] = Path(pdf_path).name
        result['file_path'] = str(pdf_path)
        return result

    def _extract_with_fitz(self, pdf_path: str) -> Dict[str, Any]:
        doc = fitz.open(pdf_path)
        full_text = ""
        pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            full_text += text + "\n"
            pages.append({
                'text': text,
                'words': [],  # PyMuPDF doesn't extract words easily
                'bbox': page.rect,
                'tables': []  # Tables are harder with PyMuPDF, skip for now
            })
        metadata = doc.metadata
        doc.close()
        return {
            'pages': pages,
            'metadata': metadata,
            'full_text': full_text
        }

    def _get_adaptive_resolution(self, page_array: np.ndarray, initial_text: str) -> float:
        """
        Определяет оптимальное разрешение для OCR на основе качества изображения.
        
        Returns:
            Коэффициент масштабирования (2.0 - 5.0)
        """
        gray = cv2.cvtColor(page_array, cv2.COLOR_BGR2GRAY) if len(page_array.shape) == 3 else page_array
        
        # Метрики качества
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = gray.std()
        brightness = np.mean(gray)
        
        # Если мало текста без OCR - вероятно скан
        is_scan = len(initial_text.strip()) < 300
        
        # Определяем разрешение
        if blur < 50 or contrast < 30:
            # Очень плохое качество - максимальное разрешение
            resolution = 5.0
        elif blur < 100 or contrast < 50 or is_scan:
            # Плохое качество - высокое разрешение
            resolution = 4.0
        elif blur < 200 or contrast < 70:
            # Среднее качество - стандартное разрешение
            resolution = 3.0
        else:
            # Хорошее качество - минимальное разрешение
            resolution = 2.0
        
        logger.debug(f"Адаптивное разрешение: blur={blur:.1f}, contrast={contrast:.1f}, resolution={resolution}x")
        return resolution
    
    def _extract_with_ocr(self, pdf_path: str, initial_text: str = "") -> str:
        """
        Извлекает текст через OCR с адаптивной предобработкой и улучшениями.
        
        Args:
            pdf_path: Путь к PDF файлу
            initial_text: Текст, извлеченный без OCR (для оценки качества)
        """
        # Используем улучшенный метод с мультимасштабной обработкой
        return self._extract_with_multiscale_ocr(pdf_path, initial_text)
    
    def _extract_with_multiscale_ocr(self, pdf_path: str, initial_text: str = "") -> str:
        """
        Извлекает текст используя несколько разрешений и выбирает лучший результат.
        """
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Получаем базовое изображение для анализа
            base_pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            base_array = self._pixmap_to_numpy(base_pix)
            
            # Определяем адаптивное разрешение
            adaptive_resolution = self._get_adaptive_resolution(base_array, initial_text)
            
            # Для сложных случаев используем мультимасштабную обработку
            if adaptive_resolution >= 4.0:
                # Очень плохое качество - пробуем несколько разрешений
                resolutions = [3.0, 4.0, 5.0]
            elif adaptive_resolution >= 3.0:
                # Плохое качество - пробуем 2-3 разрешения
                resolutions = [2.5, 3.5, 4.0]
            else:
                # Хорошее качество - одно разрешение
                resolutions = [adaptive_resolution]
            
            page_results = []
            
            for resolution in resolutions:
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(resolution, resolution))
                    resized_array = self._pixmap_to_numpy(pix)
                    
                    # Используем итеративное улучшение
                    result_text, result_confidence = self._extract_with_iterative_improvement(
                        resized_array, initial_text, page_num
                    )
                    
                    if result_text:
                        page_results.append((result_text, result_confidence, resolution))
                        logger.debug(f"Страница {page_num}, разрешение {resolution}x: {len(result_text)} символов, confidence={result_confidence:.3f}")
                
                except Exception as e:
                    logger.warning(f"Ошибка при обработке страницы {page_num} с разрешением {resolution}x: {e}")
                    continue
            
            # Комбинируем результаты от разных разрешений
            if page_results:
                texts = [(text, conf) for text, conf, _ in page_results]
                combined_text = self.result_combiner.combine_results(texts, initial_text)
                full_text += combined_text + "\n"
                logger.info(f"Страница {page_num}: комбинировано {len(page_results)} результатов")
            else:
                logger.warning(f"Не удалось извлечь текст со страницы {page_num}")
        
        doc.close()
        return full_text
    
    def _extract_with_iterative_improvement(self, page_array: np.ndarray, initial_text: str, page_num: int) -> tuple:
        """
        Итеративно улучшает результат OCR, проверяя качество и применяя
        более агрессивную предобработку при необходимости.
        
        Returns:
            Tuple[текст, confidence]
        """
        best_text = ""
        best_quality = 0.0
        best_confidence = 0.0
        iteration = 0
        max_iterations = 3
        
        # Определяем, является ли это фото
        is_photo, photo_confidence = self.photo_preprocessor.is_photo(page_array, initial_text)
        
        if is_photo:
            # Для фото используем единый пайплайн
            preprocessed = self.photo_preprocessor.preprocess(page_array)
            if self.use_confidence_selection:
                text, confidence = self.ocr_extractor.extract_text_with_confidence(preprocessed)
            else:
                text = self.ocr_extractor.extract_text(preprocessed)
                confidence = 0.8  # Предполагаем хорошее качество для фото
            
            quality = self.quality_estimator.estimate_text_quality(text, initial_text)
            return text, confidence
        
        # Для обычных документов - итеративное улучшение
        while iteration < max_iterations:
            try:
                # Выбираем метод предобработки в зависимости от итерации
                if iteration == 0:
                    # Первая итерация - легкая предобработка
                    variants = self.light_preprocessor.get_variants(page_array)
                    if variants:
                        preprocessed = variants[0][1]
                    else:
                        preprocessed = page_array
                elif iteration == 1:
                    # Вторая итерация - универсальная предобработка с детекцией деградации
                    variants = self.universal_preprocessor.get_variants(page_array)
                    if variants:
                        preprocessed = variants[0][1]
                    else:
                        preprocessed = page_array
                else:
                    # Третья итерация - тяжелая предобработка
                    variants = self.heavy_preprocessor.get_variants(page_array)
                    if variants:
                        preprocessed = variants[0][1]
                    else:
                        preprocessed = page_array
                
                # Выполняем OCR
                if self.use_confidence_selection:
                    ocr_text, ocr_confidence = self.ocr_extractor.extract_text_with_confidence(preprocessed)
                else:
                    ocr_text = self.ocr_extractor.extract_text(preprocessed)
                    ocr_confidence = 0.7  # Предполагаем среднее качество
                
                # Оцениваем качество
                quality = self.quality_estimator.estimate_text_quality(ocr_text, initial_text)
                
                # Комбинируем confidence и quality
                combined_score = (ocr_confidence * 0.6 + quality * 0.4)
                
                if combined_score > best_quality:
                    best_text = ocr_text
                    best_quality = combined_score
                    best_confidence = ocr_confidence
                
                # Если качество хорошее, останавливаемся
                if quality > 0.7 or combined_score > 0.75:
                    logger.debug(f"Страница {page_num}, итерация {iteration}: качество достаточное (quality={quality:.3f}), останавливаемся")
                    break
                
                iteration += 1
            
            except Exception as e:
                logger.warning(f"Ошибка на итерации {iteration} для страницы {page_num}: {e}")
                iteration += 1
                continue
        
        # Если все итерации не дали хорошего результата, пробуем оригинал
        if best_quality < 0.5:
            try:
                if self.use_confidence_selection:
                    original_text, original_confidence = self.ocr_extractor.extract_text_with_confidence(page_array)
                else:
                    original_text = self.ocr_extractor.extract_text(page_array)
                    original_confidence = 0.6
                
                original_quality = self.quality_estimator.estimate_text_quality(original_text, initial_text)
                original_score = (original_confidence * 0.6 + original_quality * 0.4)
                
                if original_score > best_quality:
                    best_text = original_text
                    best_confidence = original_confidence
                    logger.debug(f"Страница {page_num}: оригинал дал лучший результат")
            except Exception as e:
                logger.warning(f"Ошибка при обработке оригинала для страницы {page_num}: {e}")
        
        return best_text, best_confidence
    
    def extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Извлекает текст из изображения (JPG, PNG и т.д.).
        
        Args:
            image_path: Путь к файлу изображения
            
        Returns:
            Словарь с извлеченными данными (аналогично PDF)
        """
        logger.debug(f"Starting image extraction for {image_path}")
        
        # Проверяем размер файла
        file_size = os.path.getsize(image_path)
        if file_size > 50 * 1024 * 1024:  # 50 MB limit для изображений
            raise ValueError(f"FILE_TOO_LARGE: {image_path} ({file_size / 1024 / 1024:.1f} MB)")
        
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        logger.debug(f"Image loaded: shape={image.shape}, dtype={image.dtype}")
        
        # Для фото всегда используем единый стабильный пайплайн
        logger.info("Применяется единый пайплайн предобработки для фото")
        preprocessed = self.photo_preprocessor.preprocess(image)
        
        # Извлечение текста через OCR
        full_text = self.ocr_extractor.extract_text(preprocessed)
        
        if not full_text or len(full_text.strip()) < 10:
            logger.warning(f"Извлечено слишком мало текста из {image_path}: {len(full_text)} символов")
        
        logger.debug(f"Extracted text length: {len(full_text)}")
        
        return {
            'pages': [{
                'text': full_text,
                'words': [],
                'bbox': None,
                'tables': []
            }],
            'metadata': {
                'source': 'image',
                'file_path': image_path,
                'file_size': file_size,
                'image_shape': image.shape,
                'preprocessing_applied': 'photo_pipeline'
            },
            'full_text': full_text
        }

    def _pixmap_to_numpy(self, pix) -> np.ndarray:
        # Convert fitz Pixmap to numpy array (BGR)
        img = pix.samples
        if pix.n < 4:
            img_fmt = np.frombuffer(img, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 3:
                # Convert RGB to BGR
                img_fmt = img_fmt[..., ::-1]
        else:
            img_fmt = np.frombuffer(img, dtype=np.uint8).reshape(pix.height, pix.width, 4)
            # Ignore alpha channel
            img_fmt = img_fmt[..., :3]
            # Convert RGB to BGR
            img_fmt = img_fmt[..., ::-1]
        return img_fmt
