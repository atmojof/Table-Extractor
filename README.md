# Table and Text Extraction Tool

A Streamlit-based web application for extracting tables and text from images using OCR and advanced table recognition techniques.

![Demo Screenshot](demo.png)

## Features

- **Multiple Input Sources**:
  - Upload images (JPG, PNG, JPEG)
  - Select from sample images
  - Capture images directly from camera

- **Extraction Modes**:
  - Table extraction with structure recognition
  - Plain text extraction (OCR only)

- **Advanced Table Recognition Engines**:
  - RapidTable (SLANet and SLANet-plus)
  - Wired Table Recognition (v1 and v2)
  - Lineless Table Recognition
  - Automatic table type detection

- **Output Options**:
  - HTML table rendering
  - Excel download (preserving merged cells)
  - Plain text download
  - Visualization of recognition boxes

- **Performance Optimization**:
  - Cached model loading
  - Processing time metrics
  - Configurable thresholds for row/column detection

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/table-text-extraction.git
   cd table-text-extraction
