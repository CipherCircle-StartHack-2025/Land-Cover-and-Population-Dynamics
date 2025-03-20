import os
import glob
import json
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import xy
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd
import pyproj
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

###############################################################################
# 1. HELPER FUNCTIONS
###############################################################################

def raster_to_points_latlon(raster, transform, src_crs, sample_step=10):
    rows, cols = raster.shape
    transformer = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    data = []
    for r in range(0, rows, sample_step):
        for c in range(0, cols, sample_step):
            val = raster[r, c]
            if np.isnan(val):
                continue
            x, y = xy(transform, r, c, offset='center')
            lon, lat = transformer.transform(x, y)
            data.append((lat, lon, val))
    df = pd.DataFrame(data, columns=['lat', 'lon', 'value'])
    return df

def load_raster(file_path, is_gpp=False):
    with rasterio.open(file_path) as src:
        arr = src.read(1).astype(float)
        transform = src.transform
        crs = src.crs
        
        if is_gpp:
            arr[arr == 65533] = np.nan
            arr[arr <= 0] = np.nan
            arr[arr > 50000] = np.nan
        else:
            arr[arr <= 0] = np.nan

    return arr, transform, crs

def resample_to_match(src_data, src_transform, src_crs,
                      dst_data, dst_transform, dst_crs,
                      method=Resampling.bilinear):
    resampled = np.zeros_like(dst_data, dtype=float)
    reproject(
        source=src_data,
        destination=resampled,
        src_transform=src_transform,
        dst_transform=dst_transform,
        src_crs=src_crs,
        dst_crs=dst_crs,
        resampling=method
    )
    return resampled

def detect_significant_change(gpp_start, gpp_end, threshold=200):
    gpp_diff = gpp_end - gpp_start
    invalid_mask = np.isnan(gpp_start) | np.isnan(gpp_end)

    change_mask = np.abs(gpp_diff) > threshold
    change_mask[invalid_mask] = False
    return change_mask

def detect_urbanization(pop_start, pop_end, gpp_change_mask, pop_threshold=50):
    pop_diff = pop_end - pop_start
    pop_increase_mask = pop_diff > pop_threshold
    min_rows = min(pop_increase_mask.shape[0], gpp_change_mask.shape[0])
    min_cols = min(pop_increase_mask.shape[1], gpp_change_mask.shape[1])
    pop_increase_cropped = pop_increase_mask[:min_rows, :min_cols]
    gpp_change_cropped = gpp_change_mask[:min_rows, :min_cols]
    hotspots = pop_increase_cropped & gpp_change_cropped
    return hotspots, pop_diff[:min_rows, :min_cols]

def raster_center(raster, transform):
    rows, cols = raster.shape
    x1, y1 = xy(transform, 0, 0)
    x2, y2 = xy(transform, rows - 1, cols - 1)
    return {'lat': (y1 + y2) / 2, 'lon': (x1 + x2) / 2}

###############################################################################
# 2. MAIN ANALYSIS
###############################################################################

def run_analysis():
    modis_folder = '/Users/timurtaepov/Desktop/eda_Data/MODIS_Gross_Primary_Production_GPP/'
    pop_folder = '/Users/timurtaepov/Desktop/eda_Data/Gridded_Population_Density_Data/'
    
    gpp_files = sorted(glob.glob(os.path.join(modis_folder, '*GP.tif')))
    pop_files = sorted(glob.glob(os.path.join(pop_folder, '*Pop_*.tif')))
    
    gpp_2010_path = next((f for f in gpp_files if os.path.basename(f).split('_')[0] == '2010'), None)
    gpp_2020_path = next((f for f in gpp_files if os.path.basename(f).split('_')[0] == '2020'), None)
    
    pop_2010_path = next((f for f in pop_files if os.path.basename(f).split('_')[2].split('.')[0] == '2010'), None)
    pop_2015_path = next((f for f in pop_files if os.path.basename(f).split('_')[2].split('.')[0] == '2015'), None)
    pop_2020_path = next((f for f in pop_files if os.path.basename(f).split('_')[2].split('.')[0] == '2020'), None)
    
    if not (gpp_2010_path and gpp_2020_path and pop_2010_path and pop_2015_path and pop_2020_path):
        raise ValueError("Missing one or more required raster files.")
    
    gpp_2010, gpp_2010_transform, gpp_2010_crs = load_raster(gpp_2010_path, is_gpp=True)
    gpp_2020, gpp_2020_transform, gpp_2020_crs = load_raster(gpp_2020_path, is_gpp=True)
    
    pop_2010, pop_2010_transform, pop_2010_crs = load_raster(pop_2010_path, is_gpp=False)
    pop_2015, pop_2015_transform, pop_2015_crs = load_raster(pop_2015_path, is_gpp=False)
    pop_2020, pop_2020_transform, pop_2020_crs = load_raster(pop_2020_path, is_gpp=False)
    
    # Resample population if needed
    if gpp_2010_crs != pop_2010_crs or gpp_2010.shape != pop_2010.shape:
        pop_2010 = resample_to_match(pop_2010, pop_2010_transform, pop_2010_crs,
                                     gpp_2010, gpp_2010_transform, gpp_2010_crs,
                                     Resampling.bilinear)
    if gpp_2010_crs != pop_2015_crs or gpp_2010.shape != pop_2015.shape:
        pop_2015 = resample_to_match(pop_2015, pop_2015_transform, pop_2015_crs,
                                     gpp_2010, gpp_2010_transform, gpp_2010_crs,
                                     Resampling.bilinear)
    if gpp_2010_crs != pop_2020_crs or gpp_2010.shape != pop_2020.shape:
        pop_2020 = resample_to_match(pop_2020, pop_2020_transform, pop_2020_crs,
                                     gpp_2010, gpp_2010_transform, gpp_2010_crs,
                                     Resampling.bilinear)
    
    GPP_THRESHOLD = 200
    gpp_change_mask = detect_significant_change(gpp_2010, gpp_2020, threshold=GPP_THRESHOLD)
    POP_THRESHOLD = 50
    hotspots, pop_diff = detect_urbanization(pop_2010, pop_2020, gpp_change_mask, pop_threshold=POP_THRESHOLD)
    gpp_diff = gpp_2020 - gpp_2010
    
    return {
        'pop_2010': pop_2010,
        'pop_2015': pop_2015,
        'pop_2020': pop_2020,
        'pop_diff': pop_diff,
        'gpp_diff': gpp_diff,
        'hotspots': hotspots,
        'transform': gpp_2010_transform,
        'crs': gpp_2010_crs
    }

###############################################################################
# 3. LOAD RESULTS
###############################################################################
results = run_analysis()
pop_2010 = results["pop_2010"]
pop_2015 = results["pop_2015"]
pop_2020 = results["pop_2020"]
pop_diff = results["pop_diff"]
gpp_diff = results["gpp_diff"]
hotspots = results["hotspots"]
transform_used = results["transform"]
crs_used = results["crs"]

###############################################################################
# 4. STATIC FIGURES FOR GPP & HOTSPOTS
###############################################################################
df_gpp_diff = raster_to_points_latlon(gpp_diff, transform_used, crs_used, sample_step=10)
df_gpp_diff = df_gpp_diff.dropna(subset=["value"])
df_gpp_diff = df_gpp_diff[df_gpp_diff["value"] > 0]

df_hotspots = raster_to_points_latlon(hotspots.astype(float), transform_used, crs_used, sample_step=10)
df_hotspots = df_hotspots.dropna(subset=["value"])
df_hotspots = df_hotspots[df_hotspots["value"] > 0]

fig_gpp_diff = px.density_map(
    df_gpp_diff, lat="lat", lon="lon", z="value",
    radius=10,
    center={"lat": 17, "lon": -11},
    zoom=6,
    map_style="open-street-map",
    title="GPP Difference (2010→2020)"
)
fig_gpp_diff.update_layout(width=1800, height=800)

fig_hotspots = px.density_map(
    df_hotspots, lat="lat", lon="lon", z="value",
    radius=10,
    center={"lat": 17, "lon": -11},
    zoom=6,
    map_style="open-street-map",
    title="Urbanization/Deforestation Hotspots"
)
fig_hotspots.update_layout(width=1800, height=800)

###############################################################################
# 5. BUILD DYNAMIC DICTIONARIES FOR POP & GPP
###############################################################################
df_pop_2010 = raster_to_points_latlon(pop_2010, transform_used, crs_used, sample_step=10)
df_pop_2010 = df_pop_2010.dropna(subset=["value"])
df_pop_2010 = df_pop_2010[df_pop_2010["value"] > 0]

df_pop_2015 = raster_to_points_latlon(pop_2015, transform_used, crs_used, sample_step=10)
df_pop_2015 = df_pop_2015.dropna(subset=["value"])
df_pop_2015 = df_pop_2015[df_pop_2015["value"] > 0]

df_pop_2020 = raster_to_points_latlon(pop_2020, transform_used, crs_used, sample_step=10)
df_pop_2020 = df_pop_2020.dropna(subset=["value"])
df_pop_2020 = df_pop_2020[df_pop_2020["value"] > 0]

pop_years_dict = {
    2010: df_pop_2010,
    2015: df_pop_2015,
    2020: df_pop_2020
}

# GPP timeseries
gpp_timeseries_dict = {}
modis_folder = "/Users/timurtaepov/Desktop/eda_Data/MODIS_Gross_Primary_Production_GPP/"
gpp_years = range(2010, 2024)
for yr in gpp_years:
    file_path = os.path.join(modis_folder, f"{yr}_GP.tif")
    if os.path.exists(file_path):
        arr, transf, crs_ = load_raster(file_path, is_gpp=True)
        df = raster_to_points_latlon(arr, transf, crs_, sample_step=10)
        df = df.dropna(subset=["value"])
        df = df[df["value"] > 0]
        gpp_timeseries_dict[yr] = df

###############################################################################
# 6. PRECIPITATION BY DISTRICT
###############################################################################
def compute_precip_by_district(tif_path, shp_path):
    """
    For each district, mask the raster, compute mean precipitation.
    Returns a GeoDataFrame with a 'precip' column.
    """
    gdf = gpd.read_file(shp_path)
    with rasterio.open(tif_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        means = []
        for idx, row in gdf.iterrows():
            geom = [row.geometry]
            out_data, _ = mask(src, geom, crop=True)
            out_data = out_data.astype(float)
            if src.nodata is not None:
                out_data[out_data == src.nodata] = np.nan
            out_data[out_data == 0] = np.nan
            mean_val = np.nanmean(out_data)
            means.append(mean_val)
        gdf["precip"] = means
    return gdf

data_dir = "/Users/timurtaepov/Desktop/eda_Data/Climate_Precipitation_Data"
shp_file = "/Users/timurtaepov/Desktop/eda_Data/Admin_layers/Assaba_Districts_layer.shp"

precip_choro_dict = {}
tif_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".tif")])
for tif in tif_files:
    try:
        year = int(tif[:4])
    except:
        continue
    file_path = os.path.join(data_dir, tif)
    gdf_year = compute_precip_by_district(file_path, shp_file)
    # Convert any Timestamp columns to string
    for col in gdf_year.columns:
        if ptypes.is_datetime64_any_dtype(gdf_year[col]):
            gdf_year[col] = gdf_year[col].astype(str)
    # Reproject to EPSG:4326
    with rasterio.open(file_path) as ds:
        if gdf_year.crs != "EPSG:4326":
            gdf_year = gdf_year.to_crs(epsg=4326)
    # Unique ID
    if "ADM3_EN" not in gdf_year.columns:
        gdf_year["ADM3_EN"] = gdf_year.index.astype(str)
    precip_choro_dict[year] = gdf_year

if precip_choro_dict:
    precip_years = sorted(precip_choro_dict.keys())
else:
    precip_years = [2010]

###############################################################################
# 7. BUILD DASH APP
###############################################################################
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Assaba Region EDA Dashboard", className="text-center text-primary mb-4"), width=12)
    ]),
    dcc.Tabs([
        dcc.Tab(label="Population Over Time", children=[
            html.Div([
                html.H3("Population Over Time"),
                dcc.Slider(
                    id='pop-year-slider',
                    min=2010,
                    max=2020,
                    step=5,
                    value=2010,
                    marks={2010: '2010', 2015: '2015', 2020: '2020'}
                ),
                dcc.Graph(id='pop-year-graph', style={'width': '1800px', 'height': '800px'}),
                html.Div("Population density for the selected year. Only pixels >0 are shown.",
                         style={'marginTop': '10px', 'fontStyle': 'italic'})
            ])
        ]),
        dcc.Tab(label="GPP Trend", children=[
            html.Div([
                html.H3("GPP Trend Over Time"),
                dcc.Slider(
                    id='gpp-year-slider',
                    min=min(gpp_timeseries_dict.keys()),
                    max=max(gpp_timeseries_dict.keys()),
                    step=1,
                    value=min(gpp_timeseries_dict.keys()),
                    marks={yr: str(yr) for yr in sorted(gpp_timeseries_dict.keys())}
                ),
                dcc.Graph(id='gpp-year-graph', style={'width': '1800px', 'height': '800px'}),
                html.Div("Gross Primary Production (GPP) for the selected year. Only positive GPP shown.",
                         style={'marginTop': '10px', 'fontStyle': 'italic'})
            ])
        ]),
        dcc.Tab(label="Hotspots", children=[
            html.Div([
                dcc.Graph(figure=fig_hotspots, style={'width': '1800px', 'height': '800px'}),
                html.Div("Hotspots: where population increased >50 and GPP changed >200 from 2010 to 2020.",
                         style={'marginTop': '10px', 'fontStyle': 'italic'})
            ])
        ]),
        dcc.Tab(label="GPP Difference (Static)", children=[
            html.Div([
                dcc.Graph(figure=fig_gpp_diff, style={'width': '1800px', 'height': '800px'}),
                html.Div("Static difference in GPP (2020 - 2010).",
                         style={'marginTop': '10px', 'fontStyle': 'italic'})
            ])
        ]),
        dcc.Tab(label="Precipitation", children=[
            html.Div([
                html.H3("Precipitation Over Time (District-level)"),
                dcc.Slider(
                    id='precip-year-slider',
                    min=min(precip_years),
                    max=max(precip_years),
                    step=1,
                    value=min(precip_years),
                    marks={yr: str(yr) for yr in precip_years}
                ),
                dcc.Graph(id='precip-year-graph', style={'width': '1800px', 'height': '800px'}),
                html.Div("District-level precipitation: each district is colored by its mean precipitation for that year.",
                         style={'marginTop': '10px', 'fontStyle': 'italic'})
            ])
        ])
    ])
], fluid=True)

###############################################################################
# 8. DASH CALLBACKS
###############################################################################
@app.callback(
    Output('pop-year-graph', 'figure'),
    Input('pop-year-slider', 'value')
)
def update_population_map(selected_year):
    if selected_year not in pop_years_dict:
        return px.scatter_mapbox()
    df = pop_years_dict[selected_year]
    fig = px.density_map(
        df, lat='lat', lon='lon', z='value',
        radius=10,
        center={"lat": 17, "lon": -11},
        zoom=6,
        map_style="open-street-map",
        title=f"Population for {selected_year}"
    )
    fig.update_layout(width=1800, height=800)
    return fig

@app.callback(
    Output('gpp-year-graph', 'figure'),
    Input('gpp-year-slider', 'value')
)
def update_gpp_map(selected_year):
    if selected_year not in gpp_timeseries_dict:
        return px.scatter_mapbox()
    df = gpp_timeseries_dict[selected_year]
    fig = px.density_map(
        df, lat='lat', lon='lon', z='value',
        radius=10,
        center={"lat": 17, "lon": -11},
        zoom=6,
        map_style="open-street-map",
        title=f"GPP for {selected_year}"
    )
    fig.update_layout(width=1800, height=800)
    return fig

@app.callback(
    Output('precip-year-graph', 'figure'),
    Input('precip-year-slider', 'value')
)
def update_precip_map(selected_year):
    if selected_year not in precip_choro_dict:
        return px.scatter_mapbox()

    gdf = precip_choro_dict[selected_year]
    # Convert to GeoJSON
    gdf_json = json.loads(gdf.to_json())

    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf_json,
        locations="ADM3_EN",
        featureidkey="properties.ADM3_EN",
        color="precip",
        color_continuous_scale="Blues",
        range_color=(0, gdf["precip"].max()),
        mapbox_style="open-street-map",
        zoom=6,
        center={"lat": 17, "lon": -11},
        opacity=0.8,
        labels={"precip": "Precip (mm)"},
        title=f"Precipitation for {selected_year}"
    )
    fig.update_traces(marker_line_width=1, marker_line_color="black")
    fig.update_layout(width=1800, height=800)
    return fig

###############################################################################
# 9. RUN
###############################################################################
if __name__ == "__main__":
    app.run(debug=True)


import os
import glob
import json
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import xy
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd
import pyproj
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

###############################################################################
# 1. HELPER FUNCTIONS (for raster processing)
###############################################################################

def raster_to_points_latlon(raster, transform, src_crs, sample_step=10):
    """
    Convert a single-band, 2D raster (numpy array) to a DataFrame of lat, lon, value,
    reprojecting from src_crs to EPSG:4326.
    """
    rows, cols = raster.shape
    transformer = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    data = []
    for r in range(0, rows, sample_step):
        for c in range(0, cols, sample_step):
            val = raster[r, c]
            if np.isnan(val):
                continue
            x, y = xy(transform, r, c, offset='center')
            lon, lat = transformer.transform(x, y)
            data.append((lat, lon, val))
    df = pd.DataFrame(data, columns=['lat', 'lon', 'value'])
    return df

def load_raster(file_path, is_gpp=False):
    """
    Load a raster from a TIFF file.
    For GPP: mask known no-data (65533), zero/negative and extremely high values.
    For population: mask non-positive values.
    """
    with rasterio.open(file_path) as src:
        arr = src.read(1).astype(float)
        transform = src.transform
        crs = src.crs
        
        if is_gpp:
            arr[arr == 65533] = np.nan
            arr[arr <= 0] = np.nan
            arr[arr > 50000] = np.nan
        else:
            arr[arr <= 0] = np.nan

    return arr, transform, crs

def resample_to_match(src_data, src_transform, src_crs,
                      dst_data, dst_transform, dst_crs,
                      method=Resampling.bilinear):
    """
    Resample src_data so that its shape and resolution match dst_data.
    """
    resampled = np.zeros_like(dst_data, dtype=float)
    reproject(
        source=src_data,
        destination=resampled,
        src_transform=src_transform,
        dst_transform=dst_transform,
        src_crs=src_crs,
        dst_crs=dst_crs,
        resampling=method
    )
    return resampled

def detect_significant_change(gpp_start, gpp_end, threshold=200):
    """
    Create a boolean mask for pixels where the absolute difference in GPP is above the threshold.
    """
    gpp_diff = gpp_end - gpp_start
    invalid_mask = np.isnan(gpp_start) | np.isnan(gpp_end)
    change_mask = np.abs(gpp_diff) > threshold
    change_mask[invalid_mask] = False
    return change_mask

def detect_urbanization(pop_start, pop_end, gpp_change_mask, pop_threshold=50):
    """
    Identify hotspots where population increased more than pop_threshold and there was a significant GPP change.
    Returns the boolean hotspot mask and the population difference.
    """
    pop_diff = pop_end - pop_start
    pop_increase_mask = pop_diff > pop_threshold
    min_rows = min(pop_increase_mask.shape[0], gpp_change_mask.shape[0])
    min_cols = min(pop_increase_mask.shape[1], gpp_change_mask.shape[1])
    pop_increase_cropped = pop_increase_mask[:min_rows, :min_cols]
    gpp_change_cropped = gpp_change_mask[:min_rows, :min_cols]
    hotspots = pop_increase_cropped & gpp_change_cropped
    return hotspots, pop_diff[:min_rows, :min_cols]

def raster_center(raster, transform):
    """
    Compute the approximate center (lat/lon) of a raster.
    """
    rows, cols = raster.shape
    x1, y1 = xy(transform, 0, 0)
    x2, y2 = xy(transform, rows - 1, cols - 1)
    return {'lat': (y1 + y2) / 2, 'lon': (x1 + x2) / 2}

###############################################################################
# 2. MAIN ANALYSIS (GPP & POPULATION)
###############################################################################

def run_analysis():
    modis_folder = '/Users/timurtaepov/Desktop/eda_Data/MODIS_Gross_Primary_Production_GPP/'
    pop_folder = '/Users/timurtaepov/Desktop/eda_Data/Gridded_Population_Density_Data/'
    
    gpp_files = sorted(glob.glob(os.path.join(modis_folder, '*GP.tif')))
    pop_files = sorted(glob.glob(os.path.join(pop_folder, '*Pop_*.tif')))
    
    gpp_2010_path = next((f for f in gpp_files if os.path.basename(f).split('_')[0] == '2010'), None)
    gpp_2020_path = next((f for f in gpp_files if os.path.basename(f).split('_')[0] == '2020'), None)
    
    pop_2010_path = next((f for f in pop_files if os.path.basename(f).split('_')[2].split('.')[0] == '2010'), None)
    pop_2015_path = next((f for f in pop_files if os.path.basename(f).split('_')[2].split('.')[0] == '2015'), None)
    pop_2020_path = next((f for f in pop_files if os.path.basename(f).split('_')[2].split('.')[0] == '2020'), None)
    
    if not (gpp_2010_path and gpp_2020_path and pop_2010_path and pop_2015_path and pop_2020_path):
        raise ValueError("Missing one or more required raster files.")
    
    # Load GPP rasters
    gpp_2010, gpp_2010_transform, gpp_2010_crs = load_raster(gpp_2010_path, is_gpp=True)
    gpp_2020, gpp_2020_transform, gpp_2020_crs = load_raster(gpp_2020_path, is_gpp=True)
    
    # Load Population rasters
    pop_2010, pop_2010_transform, pop_2010_crs = load_raster(pop_2010_path, is_gpp=False)
    pop_2015, pop_2015_transform, pop_2015_crs = load_raster(pop_2015_path, is_gpp=False)
    pop_2020, pop_2020_transform, pop_2020_crs = load_raster(pop_2020_path, is_gpp=False)
    
    # Resample population if needed
    if gpp_2010_crs != pop_2010_crs or gpp_2010.shape != pop_2010.shape:
        pop_2010 = resample_to_match(pop_2010, pop_2010_transform, pop_2010_crs,
                                     gpp_2010, gpp_2010_transform, gpp_2010_crs,
                                     Resampling.bilinear)
    if gpp_2010_crs != pop_2015_crs or gpp_2010.shape != pop_2015.shape:
        pop_2015 = resample_to_match(pop_2015, pop_2015_transform, pop_2015_crs,
                                     gpp_2010, gpp_2010_transform, gpp_2010_crs,
                                     Resampling.bilinear)
    if gpp_2010_crs != pop_2020_crs or gpp_2010.shape != pop_2020.shape:
        pop_2020 = resample_to_match(pop_2020, pop_2020_transform, pop_2020_crs,
                                     gpp_2010, gpp_2010_transform, gpp_2010_crs,
                                     Resampling.bilinear)
    
    GPP_THRESHOLD = 200
    gpp_change_mask = detect_significant_change(gpp_2010, gpp_2020, threshold=GPP_THRESHOLD)
    POP_THRESHOLD = 50
    hotspots, pop_diff = detect_urbanization(pop_2010, pop_2020, gpp_change_mask, pop_threshold=POP_THRESHOLD)
    gpp_diff = gpp_2020 - gpp_2010
    
    return {
        'pop_2010': pop_2010,
        'pop_2015': pop_2015,
        'pop_2020': pop_2020,
        'pop_diff': pop_diff,
        'gpp_diff': gpp_diff,
        'hotspots': hotspots,
        'transform': gpp_2010_transform,
        'crs': gpp_2010_crs
    }

###############################################################################
# 3. LOAD RESULTS
###############################################################################
results = run_analysis()
pop_2010 = results["pop_2010"]
pop_2015 = results["pop_2015"]
pop_2020 = results["pop_2020"]
pop_diff = results["pop_diff"]
gpp_diff = results["gpp_diff"]
hotspots = results["hotspots"]
transform_used = results["transform"]
crs_used = results["crs"]

center = raster_center(pop_diff, transform_used)

###############################################################################
# 4. STATIC FIGURES FOR GPP DIFFERENCE & HOTSPOTS (for existing tabs)
###############################################################################
df_gpp_diff = raster_to_points_latlon(gpp_diff, transform_used, crs_used, sample_step=10)
df_gpp_diff = df_gpp_diff.dropna(subset=["value"])
df_gpp_diff = df_gpp_diff[df_gpp_diff["value"] > 0]

df_hotspots = raster_to_points_latlon(hotspots.astype(float), transform_used, crs_used, sample_step=10)
df_hotspots = df_hotspots.dropna(subset=["value"])
df_hotspots = df_hotspots[df_hotspots["value"] > 0]

fig_gpp_diff = px.density_map(
    df_gpp_diff, lat="lat", lon="lon", z="value",
    radius=10,
    center={"lat": 17, "lon": -11},
    zoom=6,
    map_style="open-street-map",
    title="GPP Difference (2010→2020)"
)
fig_gpp_diff.update_layout(width=1800, height=800)

fig_hotspots = px.density_map(
    df_hotspots, lat="lat", lon="lon", z="value",
    radius=10,
    center={"lat": 17, "lon": -11},
    zoom=6,
    map_style="open-street-map",
    title="Urbanization/Deforestation Hotspots"
)
fig_hotspots.update_layout(width=1800, height=800)

###############################################################################
# 5. BUILD DYNAMIC DICTIONARIES FOR POPULATION & GPP TIMESERIES
###############################################################################
df_pop_2010 = raster_to_points_latlon(pop_2010, transform_used, crs_used, sample_step=10)
df_pop_2010 = df_pop_2010.dropna(subset=["value"])
df_pop_2010 = df_pop_2010[df_pop_2010["value"] > 0]

df_pop_2015 = raster_to_points_latlon(pop_2015, transform_used, crs_used, sample_step=10)
df_pop_2015 = df_pop_2015.dropna(subset=["value"])
df_pop_2015 = df_pop_2015[df_pop_2015["value"] > 0]

df_pop_2020 = raster_to_points_latlon(pop_2020, transform_used, crs_used, sample_step=10)
df_pop_2020 = df_pop_2020.dropna(subset=["value"])
df_pop_2020 = df_pop_2020[df_pop_2020["value"] > 0]

pop_years_dict = {
    2010: df_pop_2010,
    2015: df_pop_2015,
    2020: df_pop_2020
}

gpp_timeseries_dict = {}
modis_folder = "/Users/timurtaepov/Desktop/eda_Data/MODIS_Gross_Primary_Production_GPP/"
gpp_years = range(2010, 2024)
for yr in gpp_years:
    file_path = os.path.join(modis_folder, f"{yr}_GP.tif")
    if os.path.exists(file_path):
        arr, transf, crs_ = load_raster(file_path, is_gpp=True)
        df = raster_to_points_latlon(arr, transf, crs_, sample_step=10)
        df = df.dropna(subset=["value"])
        df = df[df["value"] > 0]
        gpp_timeseries_dict[yr] = df

###############################################################################
# 6. GPP BY DISTRICT (District-level aggregation for new dashboard)
###############################################################################
def compute_gpp_by_district(tif_path, shp_path):
    """
    For each district in the shapefile, mask the GPP raster and compute the mean GPP.
    Returns a GeoDataFrame with a 'gpp' column.
    """
    gdf = gpd.read_file(shp_path)
    with rasterio.open(tif_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        means = []
        for idx, row in gdf.iterrows():
            geom = [row.geometry]
            out_data, _ = mask(src, geom, crop=True)
            out_data = out_data.astype(float)
            if src.nodata is not None:
                out_data[out_data == src.nodata] = np.nan
            # For GPP, mask out zeros and unrealistic values (threshold similar to load_raster)
            out_data[out_data <= 0] = np.nan
            out_data[out_data > 50000] = np.nan
            mean_val = np.nanmean(out_data)
            means.append(mean_val)
        gdf["gpp"] = means
    return gdf

# Build a dictionary of district-level GPP GeoDataFrames keyed by year.
shp_file = "/Users/timurtaepov/Desktop/eda_Data/Admin_layers/Assaba_Districts_layer.shp"

gpp_choro_dict = {}
for yr in gpp_years:
    file_path = os.path.join(modis_folder, f"{yr}_GP.tif")
    if os.path.exists(file_path):
        gdf_year = compute_gpp_by_district(file_path, shp_file)
        # Convert any Timestamp columns to strings to avoid JSON serialization issues
        for col in gdf_year.columns:
            if ptypes.is_datetime64_any_dtype(gdf_year[col]):
                gdf_year[col] = gdf_year[col].astype(str)
        # Reproject to EPSG:4326 for mapping
        if gdf_year.crs and gdf_year.crs.to_epsg() != 4326:
            gdf_year = gdf_year.to_crs(epsg=4326)
        # Ensure there is a unique identifier; here we assume 'ADM3_EN' is unique
        if "ADM3_EN" not in gdf_year.columns:
            gdf_year["ADM3_EN"] = gdf_year.index.astype(str)
        gpp_choro_dict[yr] = gdf_year

if gpp_choro_dict:
    gpp_choro_years = sorted(gpp_choro_dict.keys())
else:
    gpp_choro_years = [2010]

###############################################################################
# 7. BUILD THE DASH APP
###############################################################################
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Assaba Region EDA Dashboard", className="text-center text-primary mb-4"), width=12)
    ]),
    dcc.Tabs([
        # Tab 1: Population Over Time
        dbc.Tab(label="Population Over Time", children=[
            dbc.Container([
                dbc.Row([dbc.Col(html.H3("Population Over Time"), width=12)]),
                dbc.Row([dbc.Col(dcc.Slider(
                    id='pop-year-slider',
                    min=2010,
                    max=2020,
                    step=5,
                    value=2010,
                    marks={2010: '2010', 2015: '2015', 2020: '2020'}
                ), width=12)], className="mb-3"),
                dbc.Row([dbc.Col(dcc.Graph(id='pop-year-graph', style={'width': '1800px', 'height': '800px'}), width=12)]),
                dbc.Row([dbc.Col(html.Div("Population density for the selected year. Only pixels >0 are shown.",
                                            className="text-muted"), width=12)], className="mt-2")
            ])
        ]),
        # Tab 2: GPP Trend (dynamic pixel-based)
        dbc.Tab(label="GPP Trend", children=[
            dbc.Container([
                dbc.Row([dbc.Col(html.H3("GPP Trend Over Time"), width=12)]),
                dbc.Row([dbc.Col(dcc.Slider(
                    id='gpp-year-slider',
                    min=min(gpp_timeseries_dict.keys()),
                    max=max(gpp_timeseries_dict.keys()),
                    step=1,
                    value=min(gpp_timeseries_dict.keys()),
                    marks={yr: str(yr) for yr in sorted(gpp_timeseries_dict.keys())}
                ), width=12)], className="mb-3"),
                dbc.Row([dbc.Col(dcc.Graph(id='gpp-year-graph', style={'width': '1800px', 'height': '800px'}), width=12)]),
                dbc.Row([dbc.Col(html.Div("Gross Primary Production (GPP) for the selected year. Only positive GPP shown.",
                                            className="text-muted"), width=12)], className="mt-2")
            ])
        ]),
        # Tab 3: Hotspots (static)
        dbc.Tab(label="Hotspots", children=[
            dbc.Container([
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_hotspots, style={'width': '1800px', 'height': '800px'}), width=12)]),
                dbc.Row([dbc.Col(html.Div("Hotspots: where population increased >50 and GPP changed >200 from 2010 to 2020.",
                                            className="text-muted"), width=12)], className="mt-2")
            ])
        ]),
        # Tab 4: GPP Difference (static)
        dbc.Tab(label="GPP Difference (Static)", children=[
            dbc.Container([
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_gpp_diff, style={'width': '1800px', 'height': '800px'}), width=12)]),
                dbc.Row([dbc.Col(html.Div("Static difference in GPP (2020 - 2010).",
                                            className="text-muted"), width=12)], className="mt-2")
            ])
        ]),
        # Tab 5: Precipitation (existing)
        dbc.Tab(label="Precipitation", children=[
            dbc.Container([
                dbc.Row([dbc.Col(html.H3("Precipitation Over Time (District-level)"), width=12)]),
                dbc.Row([dbc.Col(html.P("Select Year:"), width=12)]),
                dbc.Row([dbc.Col(dcc.Slider(
                    id='precip-year-slider',
                    min=min(precip_years),
                    max=max(precip_years),
                    step=1,
                    value=min(precip_years),
                    marks={yr: str(yr) for yr in precip_years}
                ), width=12)], className="mb-3"),
                dbc.Row([dbc.Col(dcc.Graph(id='precip-year-graph', style={'width': '1800px', 'height': '800px'}), width=12)]),
                dbc.Row([dbc.Col(html.Div("District-level precipitation: each district is colored by its mean precipitation for that year.",
                                            className="text-muted"), width=12)], className="mt-2")
            ])
        ]),
        # NEW Tab 6: GPP by District (new GPP dashboard with satellite map)
        dbc.Tab(label="GPP by District", children=[
            dbc.Container([
                dbc.Row([dbc.Col(html.H3("GPP Over Time (District-level)"), width=12)]),
                dbc.Row([dbc.Col(html.P("Select Year:"), width=12)]),
                dbc.Row([dbc.Col(dcc.Slider(
                    id='gpp-district-year-slider',
                    min=min(gpp_choro_years),
                    max=max(gpp_choro_years),
                    step=1,
                    value=min(gpp_choro_years),
                    marks={yr: str(yr) for yr in gpp_choro_years}
                ), width=12)], className="mb-3"),
                dbc.Row([dbc.Col(dcc.Graph(id='gpp-district-graph', style={'width': '1800px', 'height': '800px'}), width=12)]),
                dbc.Row([dbc.Col(html.Div("District-level GPP: each administrative district is colored by its mean GPP for the selected year. "
                                            "The satellite basemap and black boundaries highlight the region.",
                                            className="text-muted"), width=12)], className="mt-2")
            ])
        ])
    ])
], fluid=True)

###############################################################################
# 8. DASH CALLBACKS
###############################################################################
@app.callback(
    Output('pop-year-graph', 'figure'),
    Input('pop-year-slider', 'value')
)
def update_population_map(selected_year):
    if selected_year not in pop_years_dict:
        return px.scatter_mapbox()
    df = pop_years_dict[selected_year]
    fig = px.density_map(
        df, lat='lat', lon='lon', z='value',
        radius=10,
        center={"lat": 17, "lon": -11},
        zoom=6,
        map_style="open-street-map",
        title=f"Population for {selected_year}"
    )
    fig.update_layout(width=1800, height=800)
    return fig

@app.callback(
    Output('gpp-year-graph', 'figure'),
    Input('gpp-year-slider', 'value')
)
def update_gpp_map(selected_year):
    if selected_year not in gpp_timeseries_dict:
        return px.scatter_mapbox()
    df = gpp_timeseries_dict[selected_year]
    fig = px.density_map(
        df, lat='lat', lon='lon', z='value',
        radius=10,
        center={"lat": 17, "lon": -11},
        zoom=6,
        map_style="open-street-map",
        title=f"GPP for {selected_year}"
    )
    fig.update_layout(width=1800, height=800)
    return fig

@app.callback(
    Output('precip-year-graph', 'figure'),
    Input('precip-year-slider', 'value')
)
def update_precip_map(selected_year):
    if selected_year not in precip_choro_dict:
        return px.scatter_mapbox()
    gdf = precip_choro_dict[selected_year]
    # Convert to GeoJSON (if any Timestamp columns exist, they've been converted already)
    gdf_json = json.loads(gdf.to_json())
    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf_json,
        locations="ADM3_EN",
        featureidkey="properties.ADM3_EN",
        color="precip",
        color_continuous_scale="Blues",
        range_color=(0, gdf["precip"].max()),
        mapbox_style="open-street-map",
        zoom=6,
        center={"lat": 17, "lon": -11},
        opacity=0.8,
        labels={"precip": "Precip (mm)"},
        title=f"Precipitation for {selected_year}"
    )
    fig.update_traces(marker_line_width=1, marker_line_color="black")
    fig.update_layout(width=1800, height=800)
    return fig

@app.callback(
    Output('gpp-district-graph', 'figure'),
    Input('gpp-district-year-slider', 'value')
)
def update_gpp_district_map(selected_year):
    if selected_year not in gpp_choro_dict:
        return px.scatter_mapbox()
    gdf = gpp_choro_dict[selected_year]
    # Convert GeoDataFrame to GeoJSON
    gdf_json = json.loads(gdf.to_json())
    # Create a choropleth map: each district is colored by its mean GPP (column "gpp")
    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf_json,
        locations="ADM3_EN",
        featureidkey="properties.ADM3_EN",
        color="gpp",
        color_continuous_scale="YlGn",  # choose a colormap suitable for GPP
        range_color=(0, np.nanmax(gdf["gpp"])),
        mapbox_style="satellite-streets",
        zoom=6,
        center={"lat": 17, "lon": -11},
        opacity=0.8,
        labels={"gpp": "GPP (kg_C/m²/year)"},
        title=f"GPP by District for {selected_year}"
    )
    fig.update_traces(marker_line_width=1, marker_line_color="black")
    fig.update_layout(width=1800, height=800)
    return fig

###############################################################################
# 9. RUN THE APP
###############################################################################
if __name__ == "__main__":
    app.run(debug=True)