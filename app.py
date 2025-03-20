import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import xy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pyproj

# --- Helper function to reproject raster pixel centers to EPSG:4326 ---
def raster_to_points_latlon(raster, transform, src_crs, sample_step=10):
    """
    Convert a projected raster (numpy array) to a DataFrame with lat/lon in EPSG:4326 and the pixel value.
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

# --- Other Helper Functions (your EDA functions) ---
def load_raster(file_path, is_gpp=False):
    with rasterio.open(file_path) as src:
        arr = src.read(1).astype(float)
        transform = src.transform
        crs = src.crs
        
        if is_gpp:
            # Convert known no-data
            arr[arr == 65533] = np.nan
            # Mask out zero or negative
            arr[arr <= 0] = np.nan
            # Also remove suspiciously large GPP (example threshold 50000)
            arr[arr > 50000] = np.nan
        else:
            # Convert non-positive population values to np.nan
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
    """Compute the approximate center of the raster."""
    rows, cols = raster.shape
    x1, y1 = xy(transform, 0, 0)
    x2, y2 = xy(transform, rows - 1, cols - 1)
    return {'lat': (y1 + y2) / 2, 'lon': (x1 + x2) / 2}

# --- Main Analysis Section ---
def run_analysis():
    modis_folder = '/Users/timurtaepov/Desktop/eda_Data/MODIS_Gross_Primary_Production_GPP/'
    pop_folder = '/Users/timurtaepov/Desktop/eda_Data/Gridded_Population_Density_Data/'
    
    gpp_files = sorted(glob.glob(os.path.join(modis_folder, '*GP.tif')))
    pop_files = sorted(glob.glob(os.path.join(pop_folder, '*Pop_*.tif')))
    
    # For GPP, we'll use 2010 and 2020 as our baseline (for static maps) 
    # and later build a dynamic timeseries.
    gpp_2010_path = next((f for f in gpp_files if os.path.basename(f).split('_')[0] == '2010'), None)
    gpp_2020_path = next((f for f in gpp_files if os.path.basename(f).split('_')[0] == '2020'), None)
    
    # For Population, we have 2010, 2015, and 2020
    pop_2010_path = next((f for f in pop_files if os.path.basename(f).split('_')[2].split('.')[0] == '2010'), None)
    pop_2015_path = next((f for f in pop_files if os.path.basename(f).split('_')[2].split('.')[0] == '2015'), None)
    pop_2020_path = next((f for f in pop_files if os.path.basename(f).split('_')[2].split('.')[0] == '2020'), None)
    
    if not (gpp_2010_path and gpp_2020_path and pop_2010_path and pop_2015_path and pop_2020_path):
        raise ValueError("Missing one or more required raster files (for GPP or Population).")
    
    gpp_2010, gpp_2010_transform, gpp_2010_crs = load_raster(gpp_2010_path, is_gpp=True)
    gpp_2020, gpp_2020_transform, gpp_2020_crs = load_raster(gpp_2020_path, is_gpp=True)
    pop_2010, pop_2010_transform, pop_2010_crs = load_raster(pop_2010_path, is_gpp=False)
    pop_2015, pop_2015_transform, pop_2015_crs = load_raster(pop_2015_path, is_gpp=False)
    pop_2020, pop_2020_transform, pop_2020_crs = load_raster(pop_2020_path, is_gpp=False)
    
    # Ensure population arrays match shape/CRS of gpp_2010
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
    
    # Compute differences for static maps
    GPP_THRESHOLD = 200
    gpp_change_mask = detect_significant_change(gpp_2010, gpp_2020, threshold=GPP_THRESHOLD)
    POP_THRESHOLD = 50
    hotspots, pop_diff = detect_urbanization(pop_2010, pop_2020, gpp_change_mask, pop_threshold=POP_THRESHOLD)
    gpp_diff = gpp_2020 - gpp_2010
    
    # Return all key outputs along with transform and CRS
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

results = run_analysis()

pop_2010 = results['pop_2010']
pop_2015 = results['pop_2015']
pop_2020 = results['pop_2020']
pop_diff = results['pop_diff']
gpp_diff = results['gpp_diff']
hotspots = results['hotspots']
transform_used = results['transform']
crs_used = results['crs']

center = raster_center(pop_diff, transform_used)

import plotly.express as px

# ------------------------------------------------------------------
# Convert static "gpp_diff" and "hotspots" to lat/lon and build static figures
# ------------------------------------------------------------------
df_gpp_diff = raster_to_points_latlon(gpp_diff, transform_used, crs_used, sample_step=10)
df_gpp_diff = df_gpp_diff.dropna(subset=['value'])
df_gpp_diff = df_gpp_diff[df_gpp_diff['value'] > 0]

df_hotspots = raster_to_points_latlon(hotspots.astype(float), transform_used, crs_used, sample_step=10)
df_hotspots = df_hotspots.dropna(subset=['value'])
df_hotspots = df_hotspots[df_hotspots['value'] > 0]

fig_gpp_diff = px.density_map(
    df_gpp_diff, lat='lat', lon='lon', z='value', radius=10,
    center={"lat": 17, "lon": -11}, zoom=6,
    map_style="open-street-map",
    title="GPP Difference (2010→2020)"
)
fig_gpp_diff.update_layout(width=1800, height=800)

fig_hotspots = px.density_map(
    df_hotspots, lat='lat', lon='lon', z='value', radius=10,
    center={"lat": 17, "lon": -11}, zoom=6,
    map_style="open-street-map",
    title="Urbanization/Deforestation Hotspots"
)
fig_hotspots.update_layout(width=1800, height=800)

# ------------------------------------------------------------------
# 1) Convert actual population arrays (2010, 2015, 2020) to lat/lon for dynamic slider
# ------------------------------------------------------------------
df_pop_2010 = raster_to_points_latlon(pop_2010, transform_used, crs_used, sample_step=10)
df_pop_2010 = df_pop_2010.dropna(subset=['value'])
df_pop_2010 = df_pop_2010[df_pop_2010['value'] > 0]

df_pop_2015 = raster_to_points_latlon(pop_2015, transform_used, crs_used, sample_step=10)
df_pop_2015 = df_pop_2015.dropna(subset=['value'])
df_pop_2015 = df_pop_2015[df_pop_2015['value'] > 0]

df_pop_2020 = raster_to_points_latlon(pop_2020, transform_used, crs_used, sample_step=10)
df_pop_2020 = df_pop_2020.dropna(subset=['value'])
df_pop_2020 = df_pop_2020[df_pop_2020['value'] > 0]

pop_years_dict = {
    2010: df_pop_2010,
    2015: df_pop_2015,
    2020: df_pop_2020
}

# ------------------------------------------------------------------
# 2) Build GPP timeseries dictionary
#    We'll look for GPP files named like "YYYY_GP.tif"
# ------------------------------------------------------------------
gpp_timeseries_dict = {}
modis_folder = '/Users/timurtaepov/Desktop/eda_Data/MODIS_Gross_Primary_Production_GPP/'
gpp_years = range(2010, 2024)  # adjust as needed
for yr in gpp_years:
    file_path = os.path.join(modis_folder, f"{yr}_GP.tif")
    if os.path.exists(file_path):
        arr, transf, crs_ = load_raster(file_path, is_gpp=True)
        df = raster_to_points_latlon(arr, transf, crs_, sample_step=10)
        df = df.dropna(subset=['value'])
        df = df[df['value'] > 0]
        gpp_timeseries_dict[yr] = df

# ------------------------------------------------------------------
# 3) DASH APPLICATION
# ------------------------------------------------------------------
from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Assaba Region EDA Dashboard"),
    
    dcc.Tabs([
        # --- Tab 1: Population Over Time (dynamic) ---
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
                html.Div("This map shows the population density for the selected year, derived from WorldPop data. "
                         "Only pixels with valid population values (> 0) are displayed after resampling and reprojection "
                         "to EPSG:4326.", style={'marginTop': '10px', 'fontStyle': 'italic'})
            ])
        ]),
        
        # --- Tab 2: GPP Trend (dynamic) ---
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
                html.Div("This map displays the Gross Primary Production (GPP) for the selected year, based on MODIS data. "
                         "The data is reprojected to EPSG:4326 and only pixels with valid GPP values are shown.", 
                         style={'marginTop': '10px', 'fontStyle': 'italic'})
            ])
        ]),
        
        # --- Tab 3: Hotspots (static) ---
        dcc.Tab(label="Hotspots", children=[
            html.Div([
                dcc.Graph(figure=fig_hotspots, style={'width': '1800px', 'height': '800px'}),
                html.Div("This map shows urbanization/deforestation hotspots identified by combining areas with significant "
                         "GPP change and a population increase above a defined threshold. The boolean hotspot mask was generated "
                         "by comparing 2010 and 2020 data.", style={'marginTop': '10px', 'fontStyle': 'italic'})
            ])
        ]),
        
        # --- Tab 4: GPP Difference (static) ---
        dcc.Tab(label="GPP Difference (Static)", children=[
            html.Div([
                dcc.Graph(figure=fig_gpp_diff, style={'width': '1800px', 'height': '800px'}),
                html.Div("This static map represents the difference in GPP between 2010 and 2020. It highlights areas with "
                         "significant changes in gross primary production, calculated as the difference between the two years.", 
                         style={'marginTop': '10px', 'fontStyle': 'italic'})
            ])
        ])
    ])
])

# ------------------------------------------------------------------
# 4) DASH CALLBACKS
# ------------------------------------------------------------------
@app.callback(
    Output('pop-year-graph', 'figure'),
    Input('pop-year-slider', 'value')
)
def update_population_map(selected_year):
    if selected_year not in pop_years_dict:
        return px.scatter_mapbox()  # empty figure if missing
    df = pop_years_dict[selected_year]
    fig = px.density_map(
        df, lat='lat', lon='lon', z='value', radius=10,
        center={"lat": 17, "lon": -11}, zoom=6,
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
        df, lat='lat', lon='lon', z='value', radius=10,
        center={"lat": 17, "lon": -11}, zoom=6,
        map_style="open-street-map",
        title=f"GPP for {selected_year}"
    )
    fig.update_layout(width=1800, height=800)
    return fig

# ------------------------------------------------------------------
# 5) RUN THE APP
# ------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
