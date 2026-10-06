import io
import time
import math

import cv2
import numpy as np
import streamlit as st

from PIL import Image, ImageEnhance, ImageOps

from google import genai
from google.genai import types

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Document Scanner",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 46px;
    }

    h1 {
        text-align: center;
    }

    .subtitle {
        text-align: center;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.title("📄 Smart Document Scanner")

st.markdown(
    """
    <div class="subtitle">
    Scan documents, automatically straighten them, enhance them,
    extract text with Gemini OCR, and export them as PDF.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API KEY
# ============================================================

api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:

    st.error(
        "GEMINI_API_KEY is missing. "
        "Add it in Streamlit Cloud → Settings → Secrets."
    )

    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=api_key
)


# ============================================================
# MODEL
# ============================================================

MODEL = "gemini-3.8-flash"


# ============================================================
# SESSION STATE
# ============================================================

if "pages" not in st.session_state:
    st.session_state.pages = []

if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = {}

if "processed_image" not in st.session_state:
    st.session_state.processed_image = None


# ============================================================
# OCR PROMPT
# ============================================================

OCR_PROMPT = """
You are a professional OCR engine.

Extract ALL readable text from this scanned document.

IMPORTANT:

Return ONLY the text visible in the document.

Do NOT:
- describe the document
- summarize the document
- explain anything
- identify objects
- answer questions
- add comments
- add markdown
- add quotation marks
- invent missing text

OCR RULES:

1. Extract every readable word.
2. Preserve the original language.
3. Preserve spelling.
4. Preserve capitalization where possible.
5. Preserve punctuation.
6. Preserve numbers.
7. Preserve dates.
8. Preserve prices.
9. Preserve phone numbers.
10. Preserve email addresses.
11. Preserve URLs.
12. Preserve identification numbers and codes.
13. Preserve paragraph structure.
14. Preserve line breaks when visually meaningful.
15. If the document contains a table, preserve its structure
    using spaces or tabs where possible.
16. Read printed text and handwriting when readable.
17. Do not guess unclear characters.
18. If a word is genuinely unreadable, do not invent it.
19. If there is no readable text, return:

NO TEXT FOUND

Return ONLY the extracted OCR text.
"""


# ============================================================
# IMAGE HELPERS
# ============================================================

def pil_to_cv2(image):
    """
    PIL RGB -> OpenCV BGR
    """

    image = np.array(image)

    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )


def cv2_to_pil(image):
    """
    OpenCV BGR -> PIL RGB
    """

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(image)


# ============================================================
# RESIZE
# ============================================================

def resize_for_processing(image, max_width=1800):
    """
    Resize very large images while maintaining aspect ratio.
    """

    h, w = image.shape[:2]

    if w <= max_width:
        return image

    ratio = max_width / float(w)

    new_width = int(w * ratio)
    new_height = int(h * ratio)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# FOUR POINT ORDER
# ============================================================

def order_points(points):
    """
    Arrange four points as:
    top-left
    top-right
    bottom-right
    bottom-left
    """

    points = np.array(
        points,
        dtype=np.float32
    )

    result = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    sums = points.sum(axis=1)
    differences = np.diff(
        points,
        axis=1
    ).reshape(-1)

    result[0] = points[np.argmin(sums)]
    result[2] = points[np.argmax(sums)]

    result[1] = points[
        np.argmin(differences)
    ]

    result[3] = points[
        np.argmax(differences)
    ]

    return result


# ============================================================
# FOUR POINT PERSPECTIVE TRANSFORM
# ============================================================

def four_point_transform(image, points):

    rect = order_points(points)

    top_left, top_right, bottom_right, bottom_left = rect

    width_top = np.linalg.norm(
        top_right - top_left
    )

    width_bottom = np.linalg.norm(
        bottom_right - bottom_left
    )

    max_width = int(
        max(
            width_top,
            width_bottom
        )
    )

    height_right = np.linalg.norm(
        bottom_right - top_right
    )

    height_left = np.linalg.norm(
        bottom_left - top_left
    )

    max_height = int(
        max(
            height_right,
            height_left
        )
    )

    if max_width < 10 or max_height < 10:
        return image

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (
            max_width,
            max_height,
        ),
    )

    return warped


# ============================================================
# DOCUMENT DETECTION
# ============================================================

def detect_document(image):

    original = image.copy()

    image = resize_for_processing(
        image,
        max_width=1800
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Blur reduces camera noise.
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Detect edges.
    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    # Close gaps in document edges.
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = (
        image.shape[0]
        * image.shape[1]
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < image_area * 0.15:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approximation) == 4:

            candidates.append(
                (
                    area,
                    approximation.reshape(
                        4,
                        2
                    )
                )
            )

    # Largest four-sided contour.
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    if not candidates:

        return original, False

    points = candidates[0][1]

    # Scale points back to original image.
    scale_x = (
        original.shape[1]
        / image.shape[1]
    )

    scale_y = (
        original.shape[0]
        / image.shape[0]
    )

    points = points.astype(
        np.float32
    )

    points[:, 0] *= scale_x
    points[:, 1] *= scale_y

    scanned = four_point_transform(
        original,
        points
    )

    return scanned, True


# ============================================================
# SCAN ENHANCEMENT
# ============================================================

def enhance_scan(image, mode="Color"):

    if mode == "Color":

        # Mild enhancement.
        pil = cv2_to_pil(image)

        pil = ImageEnhance.Contrast(
            pil
        ).enhance(1.08)

        pil = ImageEnhance.Sharpness(
            pil
        ).enhance(1.15)

        return pil_to_cv2(pil)


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )


# ============================================================
# IMAGE -> GEMINI PART
# ============================================================

def image_part(image):

    if len(image.shape) == 2:

        pil = Image.fromarray(
            image
        ).convert("RGB")

    else:

        pil = cv2_to_pil(
            image
        )

    buffer = io.BytesIO()

    pil.save(
        buffer,
        format="JPEG",
        quality=95,
    )

    return types.Part.from_bytes(
        data=buffer.getvalue(),
        mime_type="image/jpeg",
    )


# ============================================================
# GEMINI OCR
# ============================================================

def run_ocr(image):

    last_error = None

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    image_part(image),
                    OCR_PROMPT,
                ],
            )

            if not response.text:

                raise RuntimeError(
                    "Gemini returned an empty OCR response."
                )

            return response.text.strip()

        except Exception as e:

            last_error = e

            message = str(e).upper()

            # Temporary overload.
            if (
                "503" in message
                or "UNAVAILABLE" in message
            ):

                if attempt < 2:

                    wait = 3 * (
                        attempt + 1
                    )

                    time.sleep(
                        wait
                    )

                    continue

            # Rate limit / quota.
            if (
                "429" in message
                or "RESOURCE_EXHAUSTED" in message
                or "QUOTA" in message
            ):

                raise RuntimeError(
                    "Gemini API quota/rate limit exceeded.\n\n"
                    + str(e)
                )

            raise RuntimeError(
                str(e)
            )

    raise RuntimeError(
        f"OCR failed after retries:\n{last_error}"
    )


# ============================================================
# IMAGE TO JPEG BYTES
# ============================================================

def image_to_jpeg_bytes(image):

    if len(image.shape) == 2:

        pil = Image.fromarray(
            image
        )

    else:

        pil = cv2_to_pil(
            image
        )

    buffer = io.BytesIO()

    pil.save(
        buffer,
        format="JPEG",
        quality=95,
    )

    return buffer.getvalue()


# ============================================================
# CREATE PDF
# ============================================================

def create_pdf(images):

    output = io.BytesIO()

    pdf = canvas.Canvas(
        output
    )

    for image in images:

        pil = cv2_to_pil(
            image
        )

        width, height = pil.size

        page_width = 595
        page_height = 842

        scale = min(
            page_width / width,
            page_height / height
        )

        draw_width = width * scale
        draw_height = height * scale

        x = (
            page_width
            - draw_width
        ) / 2

        y = (
            page_height
            - draw_height
        ) / 2

        img_buffer = io.BytesIO()

        pil.save(
            img_buffer,
            format="JPEG",
            quality=95,
        )

        img_buffer.seek(0)

        pdf.drawImage(
            ImageReader(
                img_buffer
            ),
            x,
            y,
            width=draw_width,
            height=draw_height,
            preserveAspectRatio=True,
        )

        pdf.showPage()

    pdf.save()

    output.seek(0)

    return output.getvalue()


# ============================================================
# ADD PAGE
# ============================================================

def add_page(image):

    processed, detected = detect_document(
        image
    )

    st.session_state.pages.append(
        {
            "original": image,
            "scanned": processed,
            "detected": detected,
        }
    )


# ============================================================
# INPUT AREA
# ============================================================

st.subheader("📷 Scan a document")

input_method = st.radio(
    "Input",
    [
        "Camera",
        "Upload",
    ],
    horizontal=True,
)


# ============================================================
# CAMERA
# ============================================================

if input_method == "Camera":

    camera_file = st.camera_input(
        "Take a picture of your document"
    )

    if camera_file:

        if st.button(
            "➕ Add Camera Scan"
        ):

            image = Image.open(
                camera_file
            ).convert("RGB")

            image = pil_to_cv2(
                image
            )

            add_page(
                image
            )

            st.success(
                "Page added."
            )


# ============================================================
# UPLOAD
# ============================================================

else:

    uploaded_files = st.file_uploader(
        "Upload document images",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        accept_multiple_files=True,
    )

    if uploaded_files:

        if st.button(
            "➕ Add Uploaded Pages"
        ):

            for file in uploaded_files:

                image = Image.open(
                    file
                ).convert("RGB")

                image = pil_to_cv2(
                    image
                )

                add_page(
                    image
                )

            st.success(
                f"{len(uploaded_files)} page(s) added."
            )


# ============================================================
# PAGE MANAGEMENT
# ============================================================

if st.session_state.pages:

    st.markdown("---")

    st.subheader(
        f"📑 Scanned Pages ({len(st.session_state.pages)})"
    )

    for index, page in enumerate(
        st.session_state.pages
    ):

        col1, col2 = st.columns(
            [5, 1]
        )

        with col1:

            pil_image = cv2_to_pil(
                page["scanned"]
            )

            st.image(
                pil_image,
                caption=(
                    f"Page {index + 1}"
                    + (
                        " — document detected"
                        if page["detected"]
                        else " — full image used"
                    )
                ),
                use_container_width=True,
            )

        with col2:

            if st.button(
                "🗑️ Remove",
                key=f"remove_{index}",
            ):

                st.session_state.pages.pop(
                    index
                )

                st.rerun()


    # ========================================================
    # SCAN SETTINGS
    # ========================================================

    st.markdown("---")

    st.subheader(
        "⚙️ Scan Settings"
    )

    enhancement = st.radio(
        "Enhancement",
        [
            "Color",
            "Black & White",
        ],
        horizontal=True,
    )


    # ========================================================
    # PROCESS CURRENT PAGES
    # ========================================================

    if st.button(
        "✨ Apply Enhancement"
    ):

        for page in st.session_state.pages:

            page["scanned"] = enhance_scan(
                page["scanned"],
                enhancement
            )

        st.success(
            "Enhancement applied."
        )

        st.rerun()


    # ========================================================
    # OCR ALL
    # ========================================================

    st.markdown("---")

    if st.button(
        "🔤 Extract Text From All Pages"
    ):

        progress = st.progress(
            0
        )

        for index, page in enumerate(
            st.session_state.pages
        ):

            with st.spinner(
                f"Reading page {index + 1}..."
            ):

                try:

                    text = run_ocr(
                        page["scanned"]
                    )

                    st.session_state.ocr_results[
                        index
                    ] = text

                except Exception as e:

                    st.error(
                        f"Page {index + 1} OCR failed: {e}"
                    )

                    st.session_state.ocr_results[
                        index
                    ] = ""

            progress.progress(
                (index + 1)
                / len(
                    st.session_state.pages
                )
            )

        st.success(
            "OCR completed."
        )


    # ========================================================
    # OCR RESULTS
    # ========================================================

    if st.session_state.ocr_results:

        st.markdown("---")

        st.subheader(
            "📝 Extracted Text"
        )

        all_text = []

        for index in range(
            len(
                st.session_state.pages
            )
        ):

            text = st.session_state.ocr_results.get(
                index,
                ""
            )

            if text:

                st.markdown(
                    f"### Page {index + 1}"
                )

                edited = st.text_area(
                    f"OCR Page {index + 1}",
                    value=text,
                    height=250,
                    key=f"ocr_edit_{index}",
                )

                all_text.append(
                    f"--- Page {index + 1} ---\n\n"
                    + edited
                )


        combined_text = "\n\n".join(
            all_text
        )


        # ====================================================
        # DOWNLOAD TEXT
        # ====================================================

        st.download_button(
            "⬇️ Download OCR Text",
            data=combined_text,
            file_name="scanned_text.txt",
            mime="text/plain",
        )


    # ========================================================
    # EXPORT
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📤 Export"
    )

    export_col1, export_col2 = st.columns(
        2
    )


    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    with export_col1:

        pdf_bytes = create_pdf(
            [
                page["scanned"]
                for page in st.session_state.pages
            ]
        )

        st.download_button(
            "📕 Download PDF",
            data=pdf_bytes,
            file_name="scanned_document.pdf",
            mime="application/pdf",
        )


    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    with export_col2:

        if st.button(
            "🗑️ Clear All Pages"
        ):

            st.session_state.pages = []
            st.session_state.ocr_results = {}

            st.rerun()
