# Transportation-Data-Analysis

This project aims to forecast the occurrence, location, and duration of delays within the Toronto Transit Commission (TTC) network by analyzing transportation data.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Data Visualizations](#data-visualizations)
- [Data Cleaning Scripts](#data-cleaning-scripts)
- [Model Data](#model-data)
- [Subway Coordinates](#subway-coordinates)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to develop predictive models that can accurately forecast delays in the TTC network. By analyzing historical data, we aim to identify patterns and factors contributing to delays, thereby enabling proactive measures to improve transit efficiency.

## Data Sources

The repository includes the following datasets:

- **bus_model_data.csv**: Contains processed data related to bus operations, used for modeling purposes.
- **subway_model_data.csv**: Contains processed data related to subway operations, used for modeling purposes.
- **subway_coor.csv**: Provides coordinates of subway stations, useful for spatial analysis.

## Data Cleaning Scripts

To ensure data quality, several Python scripts have been developed for cleaning and preprocessing:

- **cleaning_bus.py**: Cleans and preprocesses raw bus data to prepare it for analysis.
- **cleaning_streetcar.py**: Cleans and preprocesses raw streetcar data.
- **cleaning_subway.py**: Cleans and preprocesses raw subway data.

These scripts handle tasks such as removing duplicates, handling missing values, and formatting data consistently.

## Data Visualizations

The data visualizations are in the Tableau workbook 

## Model Data

The cleaned datasets (*bus_model_data.csv* and *subway_model_data.csv*) are used to train predictive models. These models aim to forecast delays based on various features extracted from the data.

## Subway Coordinates

The *subway_coor.csv* file contains latitude and longitude information for subway stations. This spatial data is crucial for mapping and geographical analysis of delay patterns.

## Usage

To utilize the data cleaning scripts:

1. Clone the repository:

   ```bash
   git clone https://github.com/lemonface88/Transportation-Data-analysis.git
   ```

2. Navigate to the repository directory:

   ```bash
   cd Transportation-Data-analysis
   ```

3. Run the desired cleaning script:

   ```bash
   python cleaning_bus.py
   ```

   Replace `cleaning_bus.py` with the appropriate script as needed.

Ensure that all necessary dependencies are installed before running the scripts. It's recommended to use a virtual environment to manage Python packages.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please fork the repository and create a pull request. For major changes, open an issue first to discuss your ideas.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---
