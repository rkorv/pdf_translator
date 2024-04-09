from typing import Union, List, Dict
import os

import fitz
from tqdm import tqdm
import numpy as np
from PIL import Image
import imageio
import io
import cv2

from src.utils import (
    parse_page,
    split_into_paragraphs,
    merge_boxes_under_label,
    filter_strange,
    remove_boxes_by_label,
    draw_translations,
)


class PDFTranslator:
    SUPPORTED_LANGUAGES = ["ru", "de", "fr", "es", "it", "pt", "tr"]

    def __init__(
        self,
        language: str,
        model: str = "madlad400",
        use_layout: bool = True,
    ):
        """
        Args:
            language (str): Target language for translation
                - ru - Russian
                - de - German
                - fr - French
                - es - Spanish
                - it - Italian
                - pt - Portuguese
                - tr - Turkish

            model (str): Model name for translation. [madlad400, nllb200, opusmt]. Default: madlad400
                - madlad400 - Google's MADLAD400 model. The largest model available with best quality.
                              3B parameters. We recommend to use >= 16GB GPU for this model.
                              https://huggingface.co/google/madlad400-3b-mt

                - nllb200 - Facebook's NLLB200 distilled model. 600M parameters. Faster model but lower quality.
                            https://huggingface.co/facebook/nllb-200-distilled-600M

                - opusmt - OPUSMT model. The smallest model available with poor quality.
                            https://github.com/Helsinki-NLP/Opus-MT. Works only with cuda.

            use_layout (bool): Layout model will detect images and formulas to exclude them from translation. Default: True
            We use pretrain from https://github.com/LynnHaDo/Document-Layout-Analysis
        """

        assert (
            language in self.SUPPORTED_LANGUAGES
        ), f"Language {language} is not supported. Use one of {self.SUPPORTED_LANGUAGES}"
        assert model in [
            "madlad400",
            "nllb200",
            "opusmt",
        ], f"Model {model} is not supported. Use one of ['madlad400', 'nllb200', 'opusmt']"

        if model == "madlad400":
            from src.translators.madlad400 import MADLAD400Translator

            self.translator = MADLAD400Translator(language)
        elif model == "nllb200":
            from src.translators.nllb200 import NLLB200Translator

            self.translator = NLLB200Translator(language)
        elif model == "opusmt":
            from src.translators.opus_mt import OpusMTTranslator

            self.translator = OpusMTTranslator(language)

        file_path = os.path.abspath(__file__)
        self.resources_path = os.path.join(os.path.dirname(file_path), "resources")

        self.font_path = os.path.join(self.resources_path, "fonts", "Geologica-Light.ttf")

        self.use_layout = use_layout
        if self.use_layout:
            from src.layout.LayoutModel import LayoutModel

            self.layout_model = LayoutModel(
                os.path.join(self.resources_path, "models"),
            )

    def translate(
        self,
        pdf: Union[str, bytes],
        show_progress: bool = True,
        pages: List[int] = None,
    ) -> Dict:
        """
        Args:
            pdf (Union[str, bytes]): Path to PDF file or PDF bytes.
            show_progress (bool): Whether to show processing progress bar.
            pages (List[int]): List of pages to translate. If None, translate all pages.
        """

        if isinstance(pdf, str):
            doc = fitz.open(pdf)
        else:
            doc = fitz.open(stream=pdf, filetype="pdf")

        if doc.page_count == 0:
            return {}

        ### Preprocessing ###

        all_page_ids, all_bboxes, all_txts, all_lines_num = [], [], [], []

        pages_range = range(doc.page_count) if pages is None else pages

        if show_progress:
            pages_range = tqdm(pages_range, desc="Preprocessing pages")

        for page_num in pages_range:
            page = doc.load_page(page_num)
            img, bboxes, txts = parse_page(page)

            if self.use_layout:
                nn_bboxes, nn_labels, _ = self.layout_model(img)
                bboxes, txts = remove_boxes_by_label(bboxes, txts, nn_bboxes, nn_labels)
                bboxes, txts = merge_boxes_under_label(bboxes, txts, nn_bboxes, nn_labels)

            result_bboxes, result_txts, res_lines_num = [], [], []
            for bbox, txt in zip(bboxes, txts):
                bbox, txt, lines_num = split_into_paragraphs(bbox, txt)
                result_bboxes.extend(bbox)
                result_txts.extend(txt)
                res_lines_num.extend(lines_num)

            bboxes, txts, res_lines_num = filter_strange(result_bboxes, result_txts, res_lines_num)

            w, h = img.shape[1], img.shape[0]
            for i in range(len(bboxes)):
                bboxes[i] = [bboxes[i][0] / w, bboxes[i][1] / h, bboxes[i][2] / w, bboxes[i][3] / h]

            all_bboxes.append(bboxes)
            all_txts.append(txts)
            all_lines_num.append(res_lines_num)
            all_page_ids.append(page_num)

        ### Translation ###

        indices, txts_combined = [], []
        for i, txt in enumerate(all_txts):
            indices += [i] * len(txt)
            txts_combined += txt

        tr_txts = self.translator(txts_combined, show_progress=show_progress)
        all_txts = [[] for _ in range(len(all_txts))]
        for i, tr_txt in zip(indices, tr_txts):
            all_txts[i].append(tr_txt)

        pages_data = {}
        for page_id, bboxes, txts, lines_num in zip(all_page_ids, all_bboxes, all_txts, all_lines_num):
            page = []
            for bbox, txt, lines_num in zip(bboxes, txts, lines_num):
                page.append(
                    {
                        "text": txt,
                        "bbox": bbox,
                        "lines": lines_num,
                    }
                )

            if page:
                pages_data[page_id] = page

        return pages_data

    def _draw_one_page_fitz(
        self,
        doc: fitz.Document,
        data: dict,
        page_num: int,
        draw_rects: bool = False,
        scale: float = 2.5,
    ) -> np.ndarray:
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        image = np.array(image)
        w, h = image.shape[1], image.shape[0]

        if page_num in data:
            bboxes = [bbox["bbox"] for bbox in data[page_num]]
            bboxes = [[bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h] for bbox in bboxes]

            txts = [bbox["text"] for bbox in data[page_num]]
            lines_num = [bbox["lines"] for bbox in data[page_num]]

            draw_img = image.copy()
            if draw_rects:
                for bbox in bboxes:
                    x0, y0, x1, y1 = bbox
                    cv2.rectangle(draw_img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

            t_img = draw_translations(image, bboxes, txts, lines_num, self.font_path)
        else:
            print(f"[WARNING] No data for page {page_num}. Drawing original page")
            draw_img = image.copy()
            t_img = image.copy()

        white = np.zeros((draw_img.shape[0], 10, 3), dtype=np.uint8)
        merged_imgs = np.concatenate([draw_img, white, t_img], axis=1)

        return merged_imgs

    def draw_one_page(
        self,
        pdf: Union[str, io.BytesIO],
        data: dict,
        page_num: int,
        draw_rects: bool = False,
        scale: float = 2.5,
    ) -> np.ndarray:
        if isinstance(pdf, str):
            doc = fitz.open(pdf)
        else:
            doc = fitz.open(stream=pdf.read(), filetype="pdf")

        return self._draw_one_page_fitz(doc, data, page_num, draw_rects, scale)

    def save_to_pdf(
        self,
        pdf: Union[str, io.BytesIO],
        data: dict,
        save_to: str,
        show_progress: bool = True,
        scale: float = 2.5,
        jpeg_quality: int = 99,
    ):
        if isinstance(pdf, str):
            doc = fitz.open(pdf)
        else:
            doc = fitz.open(stream=pdf.read(), filetype="pdf")

        new_doc = fitz.open()
        pages = range(doc.page_count)
        if show_progress:
            pages = tqdm(pages, desc="Saving to PDF")

        for page_num in pages:
            merged_imgs = self._draw_one_page_fitz(doc, data, page_num, draw_rects=False, scale=scale)

            img_byte_arr = io.BytesIO()
            _, buffer = cv2.imencode(
                ".jpg", cv2.cvtColor(merged_imgs, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            )
            img_byte_arr.write(buffer)
            img_byte_arr.seek(0)

            new_page = new_doc.new_page(width=merged_imgs.shape[1] / scale, height=merged_imgs.shape[0] / scale)

            new_page.insert_image(new_page.rect, stream=img_byte_arr.read())

        new_doc.save(save_to)
        new_doc.close()


def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Translate PDF document")
    parser.add_argument("pdf_path", type=str, help="Path to PDF file")
    parser.add_argument(
        "language", type=str, help=f"Target language for translation: {PDFTranslator.SUPPORTED_LANGUAGES}"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="madlad400",
        help="Model name for translation. [madlad400, nllb200, opusmt]. Default: madlad400.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save translated PDF. If None, will not save",
    )
    parser.add_argument(
        "--hide_progress",
        action="store_true",
        help="Whether to show processing progress bar",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=98,
        help="JPEG quality for saving PDF. Default: 98",
    )
    parser.add_argument(
        "--pages",
        type=int,
        nargs="+",
        default=None,
        help="List of pages to translate. If None, translate all pages",
    )

    return parser


if __name__ == "__main__":
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    translator = PDFTranslator(args.language, args.model)
    data = translator.translate(args.pdf_path, show_progress=(not args.hide_progress), pages=args.pages)

    save_path = args.output
    if not save_path:
        save_path = os.path.splitext(args.pdf_path)[0] + f"_{args.language}.pdf"

    translator.save_to_pdf(
        args.pdf_path, data, save_path, show_progress=(not args.hide_progress), jpeg_quality=args.jpeg_quality
    )

    print(f"Translated PDF saved to '{save_path}'")
