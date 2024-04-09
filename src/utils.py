from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import fitz


def merge_bboxes(bboxes):
    x0 = min([bbox[0] for bbox in bboxes])
    y0 = min([bbox[1] for bbox in bboxes])
    x1 = max([bbox[2] for bbox in bboxes])
    y1 = max([bbox[3] for bbox in bboxes])
    return [x0, y0, x1, y1]


def group_same_line(bboxes, txts):
    nboxes = []
    ntxts = []
    paragraph = ""
    box = bboxes[0]
    for i, txt in enumerate(txts):
        if i == 0:
            paragraph = txt
            continue

        curr_bbox_mid_y = (bboxes[i][1] + bboxes[i][3]) / 2
        curr_bbox_h = bboxes[i][3] - bboxes[i][1]
        prev_bbox_mid_y = (bboxes[i - 1][1] + bboxes[i - 1][3]) / 2
        is_same_line = abs(curr_bbox_mid_y - prev_bbox_mid_y) < curr_bbox_h * 0.5

        if not is_same_line:
            ntxts.append(paragraph)
            paragraph = txt
            nboxes.append(box)
            box = bboxes[i]
        else:
            paragraph += " " + txt
            box = merge_bboxes([box, bboxes[i]])

    ntxts.append(paragraph)
    nboxes.append(box)
    return nboxes, ntxts


def group_lines(bboxes, txts):
    txt = "\n".join(txts)
    bbox = merge_bboxes(bboxes)
    return bbox, txt


def parse_page(page):
    page_bboxes, page_txts = [], []

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    np_img = np.array(img)

    text_dict = page.get_text("dict")

    for block in text_dict["blocks"]:
        block_lines_bboxes = []
        block_lines_txts = []
        block_lines_font_sizes = []

        if "lines" not in block:
            continue

        for line in block["lines"]:
            if not line["spans"]:
                continue

            x0 = line["spans"][0]["bbox"][0]
            y0 = sum([span["bbox"][1] for span in line["spans"]]) / len(line["spans"])
            x1 = line["spans"][-1]["bbox"][2]
            y1 = sum([span["bbox"][3] for span in line["spans"]]) / len(line["spans"])
            bbox = [x0, y0, x1, y1]
            bbox = [v * 2 for v in bbox]

            txt = "".join([span["text"] for span in line["spans"]])

            font_sizes = [span["size"] for span in line["spans"]]

            if not txt.strip():
                continue

            block_lines_bboxes.append(bbox)
            block_lines_txts.append(txt)
            block_lines_font_sizes.append(font_sizes)

        if block_lines_bboxes:
            block_lines_bboxes, block_lines_txts = group_same_line(block_lines_bboxes, block_lines_txts)
            block_lines_bboxes, block_lines_txts = group_lines(block_lines_bboxes, block_lines_txts)

            page_bboxes.append(block_lines_bboxes)
            page_txts.append(block_lines_txts)

    return np_img, page_bboxes, page_txts


def merge_boxes_under_label(bboxes, txts, nn_bboxes, nn_labels, include_labels=["text"], threshold=0.7):
    rest_nn_bboxes = [nn_bboxes[i] for i, label in enumerate(nn_labels) if label in include_labels]

    merged_bboxes = []
    merged_txts = []

    merged_indexes = set()

    for nn_bbox in rest_nn_bboxes:
        temp_bboxes = []
        temp_txts = []

        for i, bbox in enumerate(bboxes):
            if i not in merged_indexes and calculate_enclosure_percentage(bbox, nn_bbox) >= threshold:
                temp_bboxes.append(bbox)
                temp_txts.append(txts[i])
                merged_indexes.add(i)

        if temp_bboxes:
            merged_bbox = merge_bboxes(temp_bboxes)
            merged_text = "\n".join(temp_txts)

            merged_bboxes.append(merged_bbox)
            merged_txts.append(merged_text)

    for i, bbox in enumerate(bboxes):
        if i not in merged_indexes:
            merged_bboxes.append(bbox)
            merged_txts.append(txts[i])

    return merged_bboxes, merged_txts


def calculate_enclosure_percentage(bbox1, bbox2):
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

    if bbox1_area == 0:
        return 0

    return intersection_area / bbox1_area


def remove_boxes_by_label(bboxes, txts, nn_bboxes, nn_labels, exclude_labels=[["formula", "picture", "table"]]):
    rest_nn_bboxes = [nn_bboxes[i] for i, label in enumerate(nn_labels) if label in exclude_labels[0]]
    rest_bboxes = []
    rest_txts = []

    for bbox, txt in zip(bboxes, txts):
        is_enclosed = sum([calculate_enclosure_percentage(bbox, nn_bbox) for nn_bbox in rest_nn_bboxes]) > 0.8
        if not is_enclosed:
            rest_bboxes.append(bbox)
            rest_txts.append(txt)

    return rest_bboxes, rest_txts


def split_into_paragraphs(bbox, text):
    source_lines = text.strip().split("\n")
    lines = [line for line in source_lines if line.strip()]

    paragraphs = []
    lines_num = []
    bboxes = []
    height = bbox[3] - bbox[1]
    y0 = bbox[1]

    current_lines_num = 0
    current_paragraph = ""

    for i in range(len(lines)):
        current_line = lines[i].rstrip()
        current_lines_num += 1

        next_line = lines[i + 1].lstrip() if i + 1 < len(lines) else ""

        if current_line.endswith("-") and next_line and next_line[0].islower():
            current_paragraph += current_line[:-1]
        elif not (current_line[-1] in ".?!" and next_line and next_line[0].isupper()):
            current_paragraph += current_line + " "
        else:
            current_paragraph += current_line
            paragraphs.append(current_paragraph)
            y1 = bbox[1] + height * (i + 1) / len(lines)
            bboxes.append([bbox[0], y0, bbox[2], y1])
            y0 = y1
            current_paragraph = ""
            lines_num.append(current_lines_num)
            current_lines_num = 0

    if current_paragraph:
        bboxes.append([bbox[0], y0, bbox[2], bbox[3]])
        paragraphs.append(current_paragraph.rstrip())
        lines_num.append(current_lines_num)

    return bboxes, paragraphs, lines_num


def filter_strange(block_lines_bboxes, block_lines_txts, res_lines_num):
    new_block_lines_bboxes = []
    new_block_lines_txts = []
    new_res_lines_num = []

    for bbox, txt, lines_num in zip(block_lines_bboxes, block_lines_txts, res_lines_num):
        if len(txt) <= 3:
            continue

        new_block_lines_bboxes.append(bbox)
        new_block_lines_txts.append(txt)
        new_res_lines_num.append(lines_num)

    return new_block_lines_bboxes, new_block_lines_txts, new_res_lines_num


def split_text_into_n_lines(text, n):
    words = text.split()
    total_length = sum(len(word) for word in words) + len(words) - 1
    ideal_length_per_line = total_length / n

    lines = []
    current_line = []
    current_line_length = 0

    for word in words:
        if current_line and current_line_length + len(word) + 1 - ideal_length_per_line / n > ideal_length_per_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_line_length = len(word)
        else:
            if current_line:
                current_line_length += len(word) + 1
            else:
                current_line_length = len(word)
            current_line.append(word)

    lines.append(" ".join(current_line))

    while len(lines) < n:
        lines.append("")

    lines = [line.rstrip() for line in lines]
    txt = "\n".join(lines)
    return txt


def draw_translations(img, bboxes, txts, lines_num, font_path, line_spacing=1.5):
    txts = [split_text_into_n_lines(txt, line_num) for txt, line_num in zip(txts, lines_num)]

    image = Image.fromarray(np.uint8(img))
    draw = ImageDraw.Draw(image)

    for bbox, txt, line_num in zip(bboxes, txts, lines_num):
        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1

        draw.rectangle([x1, y1, x2, y2], fill="white")

        font_size = box_height / line_num
        font = ImageFont.truetype(font_path, font_size)
        lines = txt.split("\n")
        total_height_with_spacing = 0

        while font_size > 5:
            line_heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines]
            max_line_height = max(line_heights)
            one_line_height = max_line_height * line_spacing
            total_height_with_spacing = one_line_height * len(lines)
            total_width = max(font.getbbox(line)[2] - font.getbbox(line)[0] for line in lines)

            if total_height_with_spacing <= box_height and total_width <= box_width:
                break

            font_size -= 1
            if font_size < 5:
                break

            font = ImageFont.truetype(font_path, font_size)

        if font_size < 5:
            continue

        current_y = y1 + (box_height - total_height_with_spacing) / 2
        for line in lines:
            draw.text((x1, current_y), line, fill="black", font=font)
            current_y += one_line_height

    return np.array(image)
