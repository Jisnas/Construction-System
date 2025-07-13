import streamlit as st
import cv2
import numpy as np
import easyocr
import re
from PIL import Image
from typing import List, Tuple, Dict

CONVERSION_FACTORS = {
    'feet': 1.0,
    'inches': 1 / 144,
    'meters': 10.764,
    'centimeters': 0.001076,
}

# Cement Company data - Prices are now associated with cement types
CEMENT_COMPANIES = {
    "UltraTech Cement": {"OPC 33": 370, "OPC 43": 380, "OPC 53": 390, "PPC": 360, "PSC": 350, "White Cement": 800, "RMC": 400},
    "Ambuja Cements": {"OPC": 385, "PPC": 370, "Composite": 365},
    "ACC Limited": {"OPC 43": 390, "OPC 53": 400, "PPC": 375, "Bulk Cement": 380, "RMC": 410},
    "Shree Cement": {"OPC": 395, "PPC": 380, "PSC": 370, "Composite": 365},
    "Dalmia Bharat Cement": {"OPC 43": 380, "OPC 53": 390, "PPC": 370, "PSC": 360, "Sulphate Resisting": 420, "Oil Well": 450},
    "Birla Corporation Limited": {"OPC": 400, "PPC": 385, "PSC": 375, "Low Alkali": 430},
    "Ramco Cements": {"OPC 43": 390, "OPC 53": 400, "PPC": 380, "PSC": 370, "Sulphate Resisting": 425, "Rapid Hardening": 440},
    "India Cements": {"OPC": 395, "PPC": 380, "PSC": 370, "SRPC": 430},
    "JK Cement": {"OPC 43": 385, "OPC 53": 395, "PPC": 375, "White Cement": 790, "Wall Putty": 250},
    "Orient Cement": {"OPC 43": 380, "OPC 53": 390, "PPC": 370},
    "HeidelbergCement India": {"OPC 43": 390, "OPC 53": 400, "PPC": 380},
    "Sanghi Industries": {"OPC 53": 395, "OPC 43": 385, "PPC": 375, "PSC": 365},
    "Penna Cement": {"OPC": 390, "PPC": 375, "PSC": 365, "Composite": 360},
    "My Home Industries": {"OPC": 395, "PPC": 380, "PSC": 370, "RMC": 400},
    "Nuvoco Vistas Corp. Ltd.": {"OPC": 400, "PPC": 385, "PSC": 375, "Composite": 370, "RMC": 405},
    "Chettinad Cement": {"OPC 43": 380, "OPC 53": 390, "PPC": 370, "SRPC": 420, "Composite": 365},
    "KCP Cement": {"OPC 43": 385, "OPC 53": 395, "PPC": 375, "PSC": 365},
    "Deccan Cements": {"OPC 43": 390, "OPC 53": 400, "PPC": 380, "SRPC": 430, "Rapid Hardening": 440},
    "Meghalaya Cements Limited": {"OPC 43": 375, "OPC 53": 385, "PPC": 365},
    "Star Cement": {"OPC 43": 380, "OPC 53": 390, "PPC": 370, "PSC": 360},
    "Anjani Portland Cement": {"OPC 53": 395, "OPC 43": 385, "PPC": 375},
    "Maha Cement": {"OPC 43": 380, "OPC 53": 390, "PPC": 370, "PSC": 360},
    "Zuari Cement": {"OPC 43": 385, "OPC 53": 395, "PPC": 375, "RMC": 405},
    "Sagar Cements": {"OPC 53": 390, "OPC 43": 380, "PPC": 370, "PSC": 360},
    "JSW Cement": {"PSC": 355, "Composite": 350, "GGBS": 340},
    "Mangalam Cement": {"OPC 43": 375, "OPC 53": 385, "PPC": 365},
    "Kesoram Industries": {"OPC": 390},
}

# All TMT bars list
TMT_BARS = {
    "Tata Steel": {"Fe 415": 55000, "Fe 500": 56000, "Fe 550": 57000, "Fe 600": 58000},
    "JSW Steel": {"Fe 415": 54000, "Fe 500": 55000, "Fe 550": 56000, "Fe 600": 57000},
    "SAIL": {"Fe 415": 53000, "Fe 500": 54000, "Fe 550": 55000, "Fe 600": 56000},
    "RINL": {"Fe 415": 52000, "Fe 500": 53000, "Fe 550": 54000, "Fe 600": 55000},
    "Essar Steel": {"Fe 415": 51000, "Fe 500": 52000, "Fe 550": 53000, "Fe 600": 54000},
}

# Paint brand list - ADDED BACK IN
PAINT_BRANDS = {
    "Asian Paints": ["Emulsion", "Distemper", "Enamel", "Weather-Resistant", "Cement", "Texture", "Metallic", "Matte"],
    "Berger Paints": ["Emulsion", "Distemper", "Enamel", "Weather-Resistant", "Cement", "Texture", "Metallic", "Matte"],
    "Nerolac Paints": ["Emulsion", "Distemper", "Enamel", "Weather-Resistant", "Cement", "Texture", "Metallic", "Matte"],
    "Dulux Paints": ["Emulsion", "Distemper", "Enamel", "Weather-Resistant", "Cement", "Texture", "Metallic", "Matte"],
    "Shalimar Paints": ["Emulsion", "Distemper", "Enamel", "Weather-Resistant", "Cement", "Texture", "Metallic", "Matte"],
}

MATERIAL_COSTS = {
    'cement': {},  # Will be populated dynamically
    'steel': {},  # Will be populated dynamically
    'bricks': {
        'Burnt Clay First Class': 7, 'Burnt Clay Second Class': 6.5, 'Burnt Clay Third Class': 6,
        'Fly Ash Bricks': 5.5, 'Concrete Bricks Solid': 6, 'Concrete Bricks Hollow': 5,
        'AAC Blocks': 12, 'Red Clay Bricks': 6, 'Sand Lime Bricks': 8, 'Engineering Bricks': 10,
        'Fire Bricks': 15, 'Perforated Bricks': 7, 'Hollow Bricks': 5.8
    },
    'sand': {
        'River Sand': 1500, 'M-Sand': 1200, 'Pit Sand': 1300, 'Crushed Stone Sand': 1100
    },
    'tiles': {
        'Ceramic': 80, 'Vitrified Glazed': 120, 'Vitrified Double Charged': 150, 'Vitrified Full Body': 180,
        'Marble': 300, 'Granite': 250, 'Kota Stone': 100, 'Sandstone': 150, 'Wooden': 400, 'Terrazzo': 200,
        'Vinyl': 60
    },
    'flooring': {
        'Marble': 300, 'Granite': 250, 'Kota Stone': 100, 'Sandstone': 150, 'Wooden': 400,
        'Vinyl': 60, 'Terrazzo': 200, 'Cement Concrete': 50  # per sq ft
    },
    'paints': {
        "Asian Paints": {"Emulsion": 15, "Distemper": 12, "Enamel": 20, "Weather-Resistant": 25, "Cement": 10, "Texture": 30, "Metallic": 40, "Matte": 18},
        "Berger Paints": {"Emulsion": 14, "Distemper": 11, "Enamel": 19, "Weather-Resistant": 24, "Cement": 9, "Texture": 29, "Metallic": 39, "Matte": 17},
        "Nerolac Paints": {"Emulsion": 13, "Distemper": 10, "Enamel": 18, "Weather-Resistant": 23, "Cement": 8, "Texture": 28, "Metallic": 38, "Matte": 16},
        "Dulux Paints": {"Emulsion": 16, "Distemper": 13, "Enamel": 21, "Weather-Resistant": 26, "Cement": 11, "Texture": 31, "Metallic": 41, "Matte": 19},
        "Shalimar Paints": {"Emulsion": 12, "Distemper": 9, "Enamel": 17, "Weather-Resistant": 22, "Cement": 7, "Texture": 27, "Metallic": 37, "Matte": 15},
    },
    'plumbing': {
        'PVC': 50, 'CPVC': 70, 'GI': 80, 'PPR': 60, 'Brass Faucets': 500,
        'Stainless Steel Fittings': 300, 'Water Tanks': 10000, 'SWR Pipes': 40
    },  # per unit/length. Modified
    'electrical': {
        'Copper Wiring': 70, 'Aluminum Wiring': 40, 'Modular Switches': 150, 'Conventional Switches': 80,
        'MCB': 200, 'RCCB': 500, 'LED Lights': 300, 'CFL Lights': 150, 'Halogen Lights': 100
    },  # per unit. Modified
    'doors': {
        'Wooden Teak': 8000, 'Wooden Sal': 7000, 'Wooden Pine': 6000, 'Flush': 4000,
        'UPVC/PVC': 5000, 'Steel': 6000
    },  # per unit
    'windows': {
        'Wooden': 6000, 'Aluminum': 5000, 'UPVC': 4500, 'Steel': 5500
    },  # per unit
    'roofing': {
        'RCC Roof': 150, 'Flat Roof': 120, 'Sloped Roof': 180, 'Metal': 200,
        'Clay Tile': 250, 'Asphalt Shingles': 160, 'Thatch': 80, 'Polycarbonate': 220,
        'Fiberglass/Plastic': 180
    },
    'aggregates': {
        'Coarse Crushed Stone': 900, 'Coarse Gravel': 800, 'Coarse Brick Aggregate': 700,
        'Fine River Sand': 1500, 'Fine M-Sand': 1200, 'Fine Pit Sand': 1300, 'Fine Crushed Stone Sand': 1100,
        'Recycled Aggregates': 600, 'Lightweight LECA': 1800, 'Lightweight Expanded Clay': 1700,
        'Lightweight Cinder': 1600
    }
}

# No need to initialize cement costs here

THUMB_RULES = {
    'cement': 0.4,  # bags per sq ft
    'sand': 0.05,  # cubic meters per sq ft
    'steel': 2.5 / 1000,  # metric tons per sq ft
    'bricks': 7.5,  # units per sq ft
    'tiles': 1.0,  # sq ft per sq ft
    'flooring': 1.0,  # sq ft per sq ft
    'paints': 1.0,  # sq ft per sq ft
    'plumbing': 1.0,  # unit/length per sq ft (example)
    'electrical': 1.0,  # unit per sq ft (example)
    'doors': 0.01,  # units per sq ft (very rough estimate)
    'windows': 0.01,  # units per sq ft (very rough estimate)
    'roofing': 1.1,  # sq ft per sq ft (more than 1 for overlap/wastage)
    'aggregates': 0.06  # cubic meters per sq ft
}

def preprocess_image(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold, image_cv

def extract_text_from_image(image):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image)
    return ' '.join([text[1] for text in results])

def find_dimensions(text) -> List[Tuple[float, float]]:
    dimension_patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:x|×|X)\s*(\d+(?:\.\d+)?)',
        r'(\d+)\s*(?:x|×|X)\s*(\d+)',
    ]

    dimensions = set()
    for pattern in dimension_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            try:
                dim1, dim2 = match.groups()
                dim1_val = float(dim1)
                dim2_val = float(dim2)
                if dim1_val > 0 and dim2_val > 0:
                    dimensions.add(tuple(sorted([dim1_val, dim2_val])))
            except:
                continue

    return list(dimensions)

def calculate_total_area(dimensions: List[Tuple[float, float]],
                         outer_length: float,
                         outer_width: float,
                         unit: str) -> Dict[str, float]:

    conv_factor = CONVERSION_FACTORS[unit]

    floor_area = sum(dim1 * dim2 for dim1, dim2 in dimensions) * conv_factor

    ceiling_area = floor_area

    total_length = sum(dim1 + dim2 for dim1, dim2 in dimensions) + outer_length + outer_width

    if unit == 'meters':
        wall_length_meters = total_length
    elif unit == 'centimeters':
        wall_length_meters = total_length / 100
    elif unit == 'feet':
        wall_length_meters = total_length * 0.3048
    else:  # inches
        wall_length_meters = total_length * 0.0254

    wall_area_meters = wall_length_meters * 3  # 3 meters height

    wall_area = wall_area_meters * 10.764

    total_area = floor_area + ceiling_area + wall_area

    return {
        'floor': floor_area,
        'ceiling': ceiling_area,
        'walls': wall_area,
        'total': total_area
    }

def calculate_material_costs(total_area: float, material_type: str, subtype=None, company=None):
    """Calculates material costs and quantity based on the selected material type and company."""
    quantity_per_sqft = THUMB_RULES[material_type]
    total_quantity = total_area * quantity_per_sqft

    # Determine the appropriate quantity string based on material type
    if material_type == 'cement':
        quantity_string = f"{total_quantity:,.2f} bags"
        cost_per_unit = CEMENT_COMPANIES[company][subtype]  # Directly get the price from the dictionary
        total_cost = total_quantity * cost_per_unit
    elif material_type == 'steel':
        quantity_string = f"{total_quantity:,.2f} metric tons"
        cost_per_unit = TMT_BARS[company][subtype]  # Directly get the price from the dictionary
        total_cost = total_quantity * cost_per_unit
    elif material_type == 'bricks':
        quantity_string = f"{total_quantity:,.2f} units"
        cost_per_unit = MATERIAL_COSTS[material_type][subtype]  # Directly get the price from the dictionary
        total_cost = total_quantity * cost_per_unit
    elif material_type == 'sand':
        quantity_string = f"{total_quantity:,.2f} cubic meters"
        cost_per_unit = MATERIAL_COSTS[material_type][subtype]  # Directly get the price from the dictionary
        total_cost = total_quantity * cost_per_unit
    elif material_type == 'tiles':
        quantity_string = f"{total_quantity:,.2f} sq ft"
        cost_per_unit = MATERIAL_COSTS[material_type][subtype]  # Directly get the price from the dictionary
        total_cost = total_quantity * cost_per_unit
    elif material_type == 'flooring':
        quantity_string = f"{total_quantity:,.2f} sq ft"
        cost_per_unit = MATERIAL_COSTS[material_type][subtype]  # Directly get the price from the dictionary
        total_cost = total_quantity * cost_per_unit
    elif material_type == 'paints':
        quantity_string = f"{total_quantity:,.2f} sq ft"
        cost_per_unit = MATERIAL_COSTS[material_type][company][subtype]  # Directly get the price from the dictionary
        total_cost = total_quantity * cost_per_unit
    elif material_type == 'plumbing':
        quantity_string = f"{total_quantity:,.2f} units"
        cost_per_unit = MATERIAL_COSTS[material_type][subtype]  # Directly get the price from the dictionary
        total_cost = total_quantity * cost_per_unit
    elif material_type == 'electrical':
        quantity_string = f"{total_quantity:,.2f} units"
        cost_per_unit = MATERIAL_COSTS[material_type][subtype]  # Directly get the price from the dictionary
        total_cost = total_quantity * cost_per_unit
    elif material_type == 'doors':
        quantity_string = f"{total_quantity:,.2f} units"
        cost_per_unit = MATERIAL_COSTS[material_type][subtype]  # Directly get the price from the dictionary
        total_cost = total_quantity * cost_per_unit
    elif material_type == 'windows':
        quantity_string = f"{total_quantity:,.2f} units"
        cost_per_unit = MATERIAL_COSTS[material_type][subtype]  # Directly get the price from the dictionary
        total_cost = total_quantity * cost_per_unit
    elif material_type == 'roofing':
        quantity_string = f"{total_quantity:,.2f} sq ft"
        cost_per_unit = MATERIAL_COSTS[material_type][subtype]  # Directly get the price from the dictionary
        total_cost = total_quantity * cost_per_unit
    elif material_type == 'aggregates':
        quantity_string = f"{total_quantity:,.2f} cubic meters"
        cost_per_unit = MATERIAL_COSTS[material_type][subtype]  # Directly get the price from the dictionary
        total_cost = total_quantity * cost_per_unit
    else:
        quantity_string = f"{total_quantity:,.2f}"
        try:
            cost_per_unit = MATERIAL_COSTS[material_type][subtype]
        except:
            cost_per_unit = 0
        total_cost = total_quantity * cost_per_unit

    return quantity_string, total_cost


def main():
    st.title("Construction Material Calculator")
    st.write("Upload a floor plan image to calculate material quantities and costs")

    unit = st.selectbox(
        "Select measurement unit of the dimensions in the image",
        options=['feet', 'inches', 'meters', 'centimeters'],
        index=0
    )

    col1, col2 = st.columns(2)
    with col1:
        outer_length = st.number_input(f"Outer Length ({unit})", min_value=0.0, value=0.0)
    with col2:
        outer_width = st.number_input(f"Outer Width ({unit})", min_value=0.0, value=0.0)

    if 'materials_selected' not in st.session_state:
        st.session_state.materials_selected = {}

    if 'calculation_results' not in st.session_state:
        st.session_state.calculation_results = {}

    uploaded_file = st.file_uploader("Choose a floor plan image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Floor Plan', use_column_width=True)

        if st.button('Calculate Square Footage'):
            try:
                with st.spinner('Processing image...'):
                    processed_image, original_cv = preprocess_image(image)
                    extracted_text = extract_text_from_image(original_cv)

                    dimensions = find_dimensions(extracted_text)

                    if dimensions:
                        areas = calculate_total_area(dimensions, outer_length, outer_width, unit)
                        total_sqft = areas['total']

                        st.session_state.total_sqft = total_sqft

                        st.success(f"Total Square Footage: {total_sqft:,.2f} sq ft")

                        with st.expander("Detailed Breakdown"):
                            st.write(f"Floor Area: {areas['floor']:,.2f} sq ft")
                            st.write(f"Ceiling Area: {areas['ceiling']:,.2f} sq ft")
                            st.write(f"Wall Area: {areas['walls']:,.2f} sq ft")

                        with st.expander("Detected Room Dimensions"):
                            for dim1, dim2 in dimensions:
                                area_in_unit = dim1 * dim2
                                area_in_sqft = area_in_unit * CONVERSION_FACTORS[unit]
                                st.text(f"{dim1:.2f} {unit} × {dim2:.2f} {unit} = {area_in_sqft:.2f} sq ft")

                        with st.expander("Show extracted text"):
                            st.text(extracted_text)

                    else:
                        st.warning("Could not detect room dimensions. Try uploading a clearer image.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

        if 'total_sqft' in st.session_state:
            st.subheader("Material Costs")

            # Cement Selection
            cement_types = sorted(list(set(
                cement_type for company_data in CEMENT_COMPANIES.values() for cement_type in company_data
            )))
            cement_type = st.selectbox("Select Cement Type", options=cement_types, key='cement_type')

            available_companies_cement = {
                company: prices[cement_type]
                for company, prices in CEMENT_COMPANIES.items()
                if cement_type in prices
            }

            if available_companies_cement:
                company_names_cement = list(available_companies_cement.keys())
                company_cement = st.selectbox(
                    "Select Cement Company",
                    options=company_names_cement,
                    key='cement_company'
                )
                st.session_state.materials_selected['cement'] = {'company': company_cement, 'subtype': cement_type}
                st.write(f"Price for {cement_type} from {company_cement}: ₹{available_companies_cement[company_cement]:,.2f} per bag")

            else:
                st.warning(f"No companies found that offer {cement_type}.")

            # Steel Selection
            steel_types = sorted(list(set(
                steel_type for company_data in TMT_BARS.values() for steel_type in company_data
            )))
            steel_type = st.selectbox("Select Steel Type", options=steel_types, key='steel_type')

            available_companies_steel = {
                company: prices[steel_type]
                for company, prices in TMT_BARS.items()
                if steel_type in prices
            }

            if available_companies_steel:
                company_names_steel = list(available_companies_steel.keys())
                company_steel = st.selectbox(
                    "Select Steel Company",
                    options=company_names_steel,
                    key='steel_company'
                )
                st.session_state.materials_selected['steel'] = {'company': company_steel, 'subtype': steel_type}
                st.write(f"Price for {steel_type} from {company_steel}: ₹{available_companies_steel[company_steel]:,.2f} per metric ton")

            else:
                st.warning(f"No companies found that offer {steel_type}.")


            # Brick Selection
            brick_types = sorted(list(MATERIAL_COSTS['bricks'].keys()))
            brick_type = st.selectbox("Select Brick Type", options=brick_types, key='brick_type')
            st.session_state.materials_selected['bricks'] = {'company': None, 'subtype': brick_type}
            st.write(f"Price for {brick_type}: ₹{MATERIAL_COSTS['bricks'][brick_type]:,.2f} per unit")  # Display price

            # Paints Selection Logic
            paints_company = st.selectbox(
                f"Select Paints Company",
                options=PAINT_BRANDS.keys(),
                key=f'paints_company'
            )

            if paints_company:  # Only show paint types if a company is selected
                paints_subtype = st.selectbox(
                    f"Select Paint Type",
                    options=PAINT_BRANDS[paints_company],
                    key=f'paints_type'
                )
                st.session_state.materials_selected['paints'] = {'company': paints_company, 'subtype': paints_subtype}
            else:
                st.session_state.materials_selected['paints'] = {'company': None, 'subtype': None}  # Ensure 'paints' is still in the dict

            for material_type, subtypes in MATERIAL_COSTS.items():
                if material_type not in ['cement', 'steel', 'bricks', 'paints']:  # Except the cement, steel and brick selection that we did above.
                    if material_type in ['doors', 'windows', 'sand', 'tiles', 'flooring', 'plumbing',
                                         'electrical', 'roofing', 'aggregates']:  # single selector
                        subtype = st.selectbox(
                            f"Select {material_type} type",
                            options=subtypes.keys(),
                            key=material_type
                        )
                        st.session_state.materials_selected[material_type] = {'company': None, 'subtype': subtype}


            if st.button('Calculate Material Costs'):
                total_sqft = st.session_state.total_sqft
                total_project_cost = 0

                # Calculate all costs
                for material_type, selection in st.session_state.materials_selected.items():
                    subtype = selection['subtype']
                    company = selection['company']

                    try:
                        # Skip calculation if subtype is None (e.g., for paints when no brand is selected)
                        if subtype is None:
                            continue

                        material_quantity, total_cost = calculate_material_costs(
                            total_sqft, material_type, subtype, company
                        )

                        st.session_state.calculation_results[material_type] = {
                            'subtype': subtype,
                            'company': company,
                            'quantity': material_quantity,
                            'cost': total_cost
                        }
                        total_project_cost += total_cost
                    except KeyError as e:
                        st.error(f"Error calculating cost for {material_type}.  Missing data for type: {subtype}")
                        continue  # Skip to the next material

                # Show results
                st.subheader("Calculation Results")
                for material_type, result in st.session_state.calculation_results.items():
                    subtype = result['subtype']
                    company = result['company']
                    display_string = f"{material_type.capitalize()}"
                    if company:
                        display_string += f" ({company} - {subtype})"
                    else:
                        display_string += f" ({subtype})"
                    st.write(display_string + ":")
                    st.write(f"  - Quantity: {result['quantity']}")
                    st.write(f"  - Total Cost: ₹{result['cost']:,.2f}")

                st.subheader(f"Total Estimated Project Cost: ₹{total_project_cost:,.2f}")

if __name__ == "__main__":
    main()
