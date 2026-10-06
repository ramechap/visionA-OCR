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
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        background-color: #FF9D00;
        color: white;
        height: 48px;
        border: none;
    }

    .stButton>button:hover {
        background-color: #E08900;
        color: white;
    }

    h1, .subtitle {
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
    Extract readable text from images using Gemini Vision.
    </p>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API KEY
# ============================================================

api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.warning(
        "⚠️ Add GEMINI_API_KEY in Streamlit Cloud → "
        "Settings → Secrets."
    )
    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=api_key)


# ============================================================
# MODEL
# ============================================================

MODEL = "gemini-flash-latest"


# ============================================================
# IMAGE CONVERSION
# ============================================================

def image_part(img):
    """
    Convert PIL image into Gemini-compatible image Part.
    """

    buf = io.BytesIO()

    img.save(
        buf,
        format="JPEG",
        quality=95,
    )

    return types.Part.from_bytes(
        data=buf.getvalue(),
        mime_type="image/jpeg",
    )


# ============================================================
# OCR PROMPT
# ============================================================

OCR_PROMPT = """
You are an OCR engine.

Extract ALL readable text from the image.

Rules:

1. Return ONLY the extracted text.
2. Do NOT describe the image.
3. Do NOT identify objects.
4. Do NOT summarize anything.
5. Do NOT add explanations.
6. Preserve the original language.
7. Preserve the original spelling as accurately as possible.
8. Preserve line breaks and paragraphs when they are visually clear.
9. Preserve numbers, punctuation, symbols, dates, prices, IDs,
   URLs, email addresses, and special characters.
10. If the image contains a table, preserve its structure as
    clearly as possible using spaces, tabs, or lines.
11. Read text from signs, labels, documents, screenshots,
    handwriting, packaging, screens, and other visible areas.
12. Do not invent text that is not visible.
13. If there is no readable text, return exactly:
    NO TEXT FOUND

Return only the OCR result.
"""


# ============================================================
# OCR FUNCTION
# ============================================================

def run_ocr(image):
    """
    Send image to Gemini and extract text.
    """

    config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        temperature=0,
    )

    errors = []

    # Try the primary model first.
    models = [
        MODEL,
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]

    # Remove duplicates while preserving order.
    models = list(dict.fromkeys(models))

    for model in models:

        for attempt in range(3):

            try:

                response = client.models.generate_content(
                    model=model,
                    contents=[
                        image_part(image),
                        OCR_PROMPT,
                    ],
                    config=config,
                )

                if response.text:

                    text = response.text.strip()

                    return text, model

                errors.append(
                    f"{model}: empty response"
                )

                break

            except Exception as e:

                error_message = str(e)
                upper_message = error_message.upper()

                # ------------------------------------------------
                # QUOTA / RATE LIMIT
                # ------------------------------------------------

                if (
                    "429" in error_message
                    or "RESOURCE_EXHAUSTED" in upper_message
                    or "QUOTA" in upper_message
                ):

                    errors.append(
                        f"{model}: quota/rate limit exhausted"
                    )

                    # Do not retry the same model.
                    break

                # ------------------------------------------------
                # TEMPORARY SERVER ERROR
                # ------------------------------------------------

                if (
                    "503" in error_message
                    or "UNAVAILABLE" in upper_message
                    or "500" in error_message
                ):

                    if attempt < 2:

                        wait = 2 * (attempt + 1)

                        time.sleep(wait)

                        continue

                # ------------------------------------------------
                # OTHER ERROR
                # ------------------------------------------------

                errors.append(
                    f"{model}: {error_message[:300]}"
                )

                break

    raise RuntimeError(
        "OCR failed.\n\n"
        + "\n".join(errors)
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


# ============================================================
# MAIN APP
# ============================================================

if uploaded:

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded
        ).convert("RGB")

    except Exception as e:

        st.error(
            f"Could not open image: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # RESIZE LARGE IMAGE
    # --------------------------------------------------------

    image.thumbnail(
        (
            4096,
            4096,
        )
    )


    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    st.image(
        image,
        caption=uploaded.name,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # OCR BUTTON
    # --------------------------------------------------------

    if st.button("🔤 Extract Text"):

        with st.spinner(
            "Extracting text..."
        ):

            try:

                extracted_text, used_model = run_ocr(
                    image
                )

                st.session_state["ocr_result"] = (
                    uploaded.name,
                    extracted_text,
                    used_model,
                )

            except Exception as e:

                error_text = str(e)

                st.error(
                    f"OCR failed: {error_text}"
                )

                if (
                    "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "quota" in error_text.lower()
                ):

                    st.warning(
                        "Your Gemini API quota or rate limit "
                        "appears to be exhausted."
                    )

                elif (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                ):

                    st.warning(
                        "Gemini is temporarily unavailable. "
                        "Please try again."
                    )


    # --------------------------------------------------------
    # OCR RESULT
    # --------------------------------------------------------

    result = st.session_state.get(
        "ocr_result"
    )

    if (
        result
        and result[0] == uploaded.name
    ):

        filename = result[0]
        extracted_text = result[1]
        used_model = result[2]

        st.success(
            f"OCR complete — model: {used_model}"
        )

        st.subheader("📄 Extracted Text")

        if extracted_text:

            st.text_area(
                "OCR Result",
                extracted_text,
                height=400,
            )

            # ------------------------------------------------
            # DOWNLOAD TXT
            # ------------------------------------------------

            st.download_button(
                "⬇️ Download TXT",
                extracted_text,
                file_name="extracted_text.txt",
                mime="text/plain",
            )

            # ------------------------------------------------
            # COPY-FRIENDLY OUTPUT
            # ------------------------------------------------

            st.markdown("### Extracted text")

            st.code(
                extracted_text,
                language=None,
            )

        else:

            st.info(
                "No readable text found."
            )
