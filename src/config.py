import os

# 基础路径设定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

# 确保输出文件夹存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据文件路径
EDGES_PATH = os.path.join(DATA_DIR, "edges")
POI_PATHS = {
    "restaurant": (os.path.join(DATA_DIR, "restaurant_poi.geojson"), 500),
    "shop": (os.path.join(DATA_DIR, "shop_poi.geojson"), 300),
    "leisure": (os.path.join(DATA_DIR, "leisure_poi.geojson"), 500),
    "tourism": (os.path.join(DATA_DIR, "tourism_poi.geojson"), 800),
    "school": (os.path.join(DATA_DIR, "school_poi.geojson"), 800)
}

# 算法配置
TARGET_COV = 0.9
CANDIDATE_NUM = 300