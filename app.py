import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd

# Import recognition engines and helper functions
from lineless_table_rec import LinelessTableRecognition
from rapid_table import RapidTable, RapidTableInput
from rapid_table.main import ModelType
from rapidocr_onnxruntime import RapidOCR
from table_cls import TableCls
from wired_table_rec import WiredTableRecognition
from utils import plot_rec_box, LoadImage, format_html, box_4_2_poly_to_box_4_1

# -----------------------------------------------------------------------------
# Advanced settings in sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("Advanced Settings")
table_engine_type = st.sidebar.selectbox(
    "Select Recognition Table Engine",
    ["auto",
     "RapidTable(SLANet)",
     "RapidTable(SLANet-plus)",
     "RapidTable(unitable)",
     "wired_table_v2",
     "wired_table_v1",
     "lineless_table"],
    index=0
)
small_box_cut_enhance = st.sidebar.checkbox("Box Cutting Enhancement", value=True)
char_ocr = st.sidebar.checkbox("Character-level OCR", value=False)
rotated_fix = st.sidebar.checkbox("Table Rotate Rec Enhancement", value=False)
col_threshold = st.sidebar.slider("Column threshold (determine same col)", 5, 100, 15, step=5)
row_threshold = st.sidebar.slider("Row threshold (determine same row)", 5, 100, 10, step=5)

# -----------------------------------------------------------------------------
# Main title and file uploader in body
# -----------------------------------------------------------------------------
st.title("📝 Image-To-Text")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -----------------------------------------------------------------------------
# Cache heavy model loading so it is only loaded once per session
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_engines():
    # Local model paths for OCR
    det_model_dir = {"mobile_det": "models/ocr/ch_PP-OCRv4_det_infer.onnx"}
    rec_model_dir = {"mobile_rec": "models/ocr/ch_PP-OCRv4_rec_infer.onnx"}
    
    # Local model path for table recognition (provided by you)
    table_rec_model_path = "models/table_rec/ch_ppstructure_mobile_v2_SLANet.onnx"
    
    # Initialize RapidTable engines (all using the local table_rec_model_path)
    rapid_table_engine = RapidTable(
        RapidTableInput(model_type=ModelType.PPSTRUCTURE_ZH.value, model_path=table_rec_model_path)
    )
    SLANet_plus_table_Engine = RapidTable(
        RapidTableInput(model_type=ModelType.SLANETPLUS.value, model_path=table_rec_model_path)
    )
    unitable_table_Engine = RapidTable(
        RapidTableInput(model_type=ModelType.UNITABLE.value, model_path=table_rec_model_path)
    )
    # Wired and Lineless engines
    wired_table_engine_v1 = WiredTableRecognition(version="v1")
    wired_table_engine_v2 = WiredTableRecognition(version="v2")
    lineless_table_engine = LinelessTableRecognition()
    table_cls = TableCls()
    
    # Build a dictionary of OCR engines using the local OCR model files.
    ocr_engine_dict = {}
    for det_model in det_model_dir.keys():
        for rec_model in rec_model_dir.keys():
            key = f"{det_model}_{rec_model}"
            ocr_engine_dict[key] = RapidOCR(
                det_model_path=det_model_dir[det_model],
                rec_model_path=rec_model_dir[rec_model]
            )
    
    return {
        "rapid_table_engine": rapid_table_engine,
        "SLANet_plus_table_Engine": SLANet_plus_table_Engine,
        "unitable_table_Engine": unitable_table_Engine,
        "wired_table_engine_v1": wired_table_engine_v1,
        "wired_table_engine_v2": wired_table_engine_v2,
        "lineless_table_engine": lineless_table_engine,
        "table_cls": table_cls,
        "ocr_engine_dict": ocr_engine_dict,
    }

engines = load_engines()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def trans_char_ocr_res(ocr_res):
    """Transform OCR results for character-level recognition."""
    word_result = []
    for res in ocr_res:
        score = res[2]
        for word_box, word in zip(res[3], res[4]):
            word_result.append([word_box, word, score])
    return word_result

def select_ocr_model(det_model, rec_model):
    """Return the OCR engine given model keys."""
    return engines["ocr_engine_dict"][f"{det_model}_{rec_model}"]

def select_table_model(img, table_engine_type, det_model, rec_model):
    """Select and return a table recognition engine based on the chosen type."""
    if table_engine_type == "RapidTable(SLANet)":
        return engines["rapid_table_engine"], table_engine_type
    elif table_engine_type == "RapidTable(SLANet-plus)":
        return engines["SLANet_plus_table_Engine"], table_engine_type
    elif table_engine_type == "RapidTable(unitable)":
        return engines["unitable_table_Engine"], table_engine_type
    elif table_engine_type == "wired_table_v1":
        return engines["wired_table_engine_v1"], table_engine_type
    elif table_engine_type == "wired_table_v2":
        return engines["wired_table_engine_v2"], table_engine_type
    elif table_engine_type == "lineless_table":
        return engines["lineless_table_engine"], table_engine_type
    elif table_engine_type == "auto":
        cls, _ = engines["table_cls"](img)
        if cls == 'wired':
            return engines["wired_table_engine_v2"], "wired_table_v2"
        return engines["lineless_table_engine"], "lineless_table"

def process_image(img_input, small_box_cut_enhance, table_engine_type, char_ocr, rotated_fix, col_threshold, row_threshold):
    # For OCR we use the mobile_det and mobile_rec models.
    det_model = "mobile_det"
    rec_model = "mobile_rec"
    img_loader = LoadImage()
    img = img_loader(img_input)
    start = time.time()
    
    table_engine, table_type = select_table_model(img, table_engine_type, det_model, rec_model)
    ocr_engine = select_ocr_model(det_model, rec_model)
    
    ocr_res, ocr_infer_elapse = ocr_engine(img, return_word_box=char_ocr)
    det_cost, cls_cost, rec_cost = ocr_infer_elapse
    if char_ocr:
        ocr_res = trans_char_ocr_res(ocr_res)
    ocr_boxes = [box_4_2_poly_to_box_4_1(ori_ocr[0]) for ori_ocr in ocr_res]
    
    if isinstance(table_engine, RapidTable):
        table_results = table_engine(img, ocr_res)
        html, polygons, table_rec_elapse = table_results.pred_html, table_results.cell_bboxes, table_results.elapse
        # Adjust polygon format to (x1, y1, x2, y2)
        polygons = [[p[0], p[1], p[4], p[5]] for p in polygons]
    elif isinstance(table_engine, (WiredTableRecognition, LinelessTableRecognition)):
        html, table_rec_elapse, polygons, _, ocr_res = table_engine(
            img, ocr_result=ocr_res,
            enhance_box_line=small_box_cut_enhance,
            rotated_fix=rotated_fix,
            col_threshold=col_threshold,
            row_threshold=row_threshold
        )
    
    sum_elapse = time.time() - start
    all_elapse = (
        f"- table_type: {table_type}\n"
        f"- table all cost: {sum_elapse:.5f}\n"
        f"- table rec cost: {table_rec_elapse:.5f}\n"
        f"- ocr cost: {det_cost + cls_cost + rec_cost:.5f}"
    )
    
    # Convert image to RGB for display
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    table_boxes_img = plot_rec_box(img.copy(), polygons)
    ocr_boxes_img = plot_rec_box(img.copy(), ocr_boxes)
    complete_html = format_html(html)
    return complete_html, table_boxes_img, ocr_boxes_img, all_elapse

# -----------------------------------------------------------------------------
# Main App Workflow
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error("Error loading image: " + str(e))
    else:
        st.image(img, caption="Uploaded Image", use_container_width=True)
        if st.button("Run Recognition"):
            with st.spinner("Processing image..."):
                complete_html, table_boxes_img, ocr_boxes_img, all_elapse = process_image(
                    img, small_box_cut_enhance, table_engine_type,
                    char_ocr, rotated_fix, col_threshold, row_threshold
                )
            st.markdown("### Recognized Table (HTML)")
            st.markdown(complete_html, unsafe_allow_html=True)
            st.markdown("### Table Recognition Boxes")
            st.image(table_boxes_img, use_container_width=True)
            st.markdown("### OCR Boxes")
            st.image(ocr_boxes_img, use_container_width=True)
            st.markdown("### Elapsed Time")
            st.text(all_elapse)
            
            # Parse the complete_html into a DataFrame and show it as a table
            try:
                df_list = pd.read_html(complete_html)
                if df_list:
                    df = df_list[0]
                    st.markdown("### Extracted Table Data")
                    st.dataframe(df)
                    # Provide a CSV download button
                    csv_data = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="table.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error("Failed to parse HTML table: " + str(e))
else:
    st.info("Please upload an image to begin.")
