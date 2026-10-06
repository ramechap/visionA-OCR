import io
import time

import streamlit as st
from PIL import Image
from google import genai
from google.genai import types


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Gemini OCR",
    page_icon="🔤",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        background-color: #FF9D00;
        color: white;
        height: 48px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #E08900;
        color: white;
    }

    h1 {
        text-align: center;
    }

    .subtitle {
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.title("🔤 Gemini OCR")

st.markdown(
    """
    <p class="subtitle">
    Extract text from images using Gemini Vision.
    </p>
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
        "Add it to Streamlit Cloud → Settings → Secrets."
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

# ============================================================
# OCR MODELS
# ============================================================

OCR_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

# ============================================================
# OCR PROMPT
# ============================================================

OCR_PROMPT = """
You are an OCR engine.

Extract ALL readable text from the image.

IMPORTANT:

- Return ONLY the text visible in the image.
- Do not describe the image.
- Do not summarize the image.
- Do not identify objects.
- Do not answer questions.
- Do not add explanations.
- Do not add markdown.
- Do not add quotation marks around the result.
- Do not invent missing text.

OCR REQUIREMENTS:

1. Extract every readable word.
2. Preserve the original language.
3. Preserve spelling as accurately as possible.
4. Preserve capitalization where possible.
5. Preserve numbers.
6. Preserve punctuation.
7. Preserve symbols.
8. Preserve dates.
9. Preserve prices.
10. Preserve phone numbers.
11. Preserve email addresses.
12. Preserve URLs.
13. Preserve IDs and codes.
14. Preserve line breaks when visually clear.
15. Preserve paragraph structure when possible.
16. If there is a table, preserve the rows and columns
    as clearly as possible using spaces or tabs.
17. Read text from documents, signs, labels, screenshots,
    receipts, forms, packaging, handwriting, and displays.
18. Do not guess text that cannot be read.
19. If there is no readable text, return exactly:

NO TEXT FOUND

Return ONLY the OCR text.
"""


# ============================================================
# IMAGE -> GEMINI PART
# ============================================================

def image_part(image):
    """
    Convert PIL image into a Gemini image Part.
    """

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=95,
    )

    return types.Part.from_bytes(
        data=buffer.getvalue(),
        mime_type="image/jpeg",
    )


# ============================================================
# OCR
# ============================================================

# ============================================================
# OCR
# ============================================================

def extract_text(image):
    """
    Try multiple Gemini models.
    If one model is temporarily overloaded,
    automatically try the next model.
    """

    errors = []

    for model in OCR_MODELS:

        for attempt in range(2):

            try:

                response = client.models.generate_content(
                    model=model,
                    contents=[
                        image_part(image),
                        OCR_PROMPT,
                    ],
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return response.text.strip(), model

            except Exception as e:

                error_text = str(e).upper()

                errors.append(
                    f"{model}: {str(e)}"
                )

                # ---------------------------------------------
                # 503 - TEMPORARY SERVER OVERLOAD
                # ---------------------------------------------

                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                ):

                    if attempt == 0:
                        time.sleep(3)
                        continue

                    # Try next model
                    break

                # ---------------------------------------------
                # 429 - QUOTA / RATE LIMIT
                # ---------------------------------------------

                if (
                    "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "QUOTA" in error_text
                ):

                    # Try next model
                    break

                # ---------------------------------------------
                # 404 - MODEL NOT AVAILABLE
                # ---------------------------------------------

                if (
                    "404" in error_text
                    or "NOT_FOUND" in error_text
                ):

                    # Try next model
                    break

                # ---------------------------------------------
                # OTHER ERROR
                # ---------------------------------------------

                raise RuntimeError(
                    f"OCR failed with {model}:\n\n{e}"
                )

    raise RuntimeError(
        "All Gemini OCR models failed.\n\n"
        + "\n\n".join(errors)
    )

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


# ============================================================
# MAIN
# ============================================================

if uploaded_file:

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as e:

        st.error(
            f"Could not open image: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # KEEP GOOD OCR RESOLUTION
    # --------------------------------------------------------

    max_dimension = 4096

    if (
        image.width > max_dimension
        or image.height > max_dimension
    ):

        image.thumbnail(
            (
                max_dimension,
                max_dimension,
            ),
            Image.Resampling.LANCZOS,
        )


    # --------------------------------------------------------
    # SHOW IMAGE
    # --------------------------------------------------------

    st.image(
        image,
        caption=uploaded_file.name,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # OCR BUTTON
    # --------------------------------------------------------

    if st.button("🔤 Extract Text"):

        with st.spinner(
            "Reading text from image..."
        ):

            try:

                extracted_text, used_model = extract_text(image)
                # Store result.
                st.session_state["ocr_text"] = (
                    uploaded_file.name,
                    extracted_text,
                    used_model,
                )

            except Exception as e:

                error_text = str(e)

                st.error(
                    f"OCR failed:\n\n{error_text}"
                )

                # ------------------------------------------------
                # FRIENDLY ERROR MESSAGES
                # ------------------------------------------------

                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                ):

                    st.warning(
                        "Gemini 3.8 Flash is temporarily "
                        "experiencing high demand. "
                        "The app automatically retried the request."
                    )

                elif (
                    "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "QUOTA" in error_text.upper()
                ):

                    st.warning(
                        "Your Gemini API quota or rate limit "
                        "has been exhausted."
                    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    result = st.session_state.get(
        "ocr_text"
    )


    if (
        result
        and result[0] == uploaded_file.name
    ):

        extracted_text = result[1]
        used_model = result[2]

        st.success(
            "OCR completed successfully."
        )
        st.caption(f"OCR model used: {used_model}")

        st.subheader(
            "📄 Extracted Text"
        )

        if extracted_text:

            # ------------------------------------------------
            # EDITABLE OCR RESULT
            # ------------------------------------------------

            edited_text = st.text_area(
                "OCR result",
                value=extracted_text,
                height=400,
            )


            # ------------------------------------------------
            # DOWNLOAD TXT
            # ------------------------------------------------

            st.download_button(
                "⬇️ Download TXT",
                data=edited_text,
                file_name="extracted_text.txt",
                mime="text/plain",
            )


            # ------------------------------------------------
            # RAW OUTPUT
            # ------------------------------------------------

            st.markdown(
                "### Extracted text"
            )

            st.code(
                edited_text,
                language=None,
            )

        else:

            st.info(
                "No readable text found."
            )
