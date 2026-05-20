import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    CRS,
    BBox
)

# =====================================================
# SENTINEL HUB CONFIG
# =====================================================

SH_CLIENT_ID = os.getenv("SH_CLIENT_ID")

SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET")

config = SHConfig()

config.sh_client_id = SH_CLIENT_ID
config.sh_client_secret = SH_CLIENT_SECRET

# =====================================================
# BASE PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

LIVE_JSON = os.path.join(
    BASE_DIR,
    "static",
    "news_data",
    "incidents.json"
)

OUTPUT_JSON = os.path.join(
    BASE_DIR,
    "static",
    "news_data",
    "historic_incidents.json"
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "static",
    "historic_images"
)

os.makedirs(IMAGE_DIR, exist_ok=True)

# =====================================================
# RGB SCRIPT
# =====================================================

RGB_SCRIPT = """
//VERSION=3

function setup() {
  return {
    input: ["B04", "B03", "B02"],
    output: { bands: 3 }
  };
}

function evaluatePixel(sample) {
  return [sample.B04, sample.B03, sample.B02];
}
"""

# =====================================================
# CREATE PLACEHOLDER IMAGE
# =====================================================

def create_placeholder(path):

    img = np.zeros(
        (512, 512, 3),
        dtype=np.uint8
    )

    plt.figure(figsize=(5, 5))

    plt.imshow(img)

    plt.axis("off")

    plt.text(

        256,
        256,

        "NO IMAGE",

        color="white",

        ha="center",

        va="center",

        fontsize=20
    )

    plt.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

# =====================================================
# CREATE BBOX
# =====================================================

def make_bbox(lat, lon):

    radius_km = 20

    d_lat = radius_km / 111

    d_lon = radius_km / (
        111 *
        math.cos(
            math.radians(lat)
        )
    )

    return (

        lon - d_lon,

        lat - d_lat,

        lon + d_lon,

        lat + d_lat
    )

# =====================================================
# FETCH IMAGE
# =====================================================

def fetch_image(
    bbox_coords,
    time_interval
):

    try:

        bbox = BBox(
            bbox=bbox_coords,
            crs=CRS.WGS84
        )

        request = SentinelHubRequest(

            evalscript=RGB_SCRIPT,

            input_data=[

                SentinelHubRequest.input_data(

                    data_collection=
                    DataCollection.SENTINEL2_L2A,

                    time_interval=time_interval,

                    maxcc=0.5
                )
            ],

            responses=[

                SentinelHubRequest.output_response(
                    "default",
                    MimeType.PNG
                )
            ],

            bbox=bbox,

            size=[512, 512],

            config=config
        )

        data = request.get_data()

        if not data:
            return None

        return data[0]

    except Exception as e:

        print("Image fetch failed:", e)

        return None

# =====================================================
# MAIN PIPELINE
# =====================================================

def generate_historic_incidents():

    print("Generating historic incidents...")

    # =================================================
    # LOAD 90-DAY INCIDENTS
    # =================================================

    if not os.path.exists(LIVE_JSON):

        print("incidents.json not found")

        return []

    with open(
        LIVE_JSON,
        "r",
        encoding="utf-8"
    ) as f:

        incidents = json.load(f)

    print(f"Loaded {len(incidents)} incidents")

    # =================================================
    # SORT OLDEST FIRST
    # =================================================

    incidents = sorted(

        incidents,

        key=lambda x:
            x.get("date", "")
    )

    # =================================================
    # TAKE FIRST 10
    # =================================================

    historic_news = incidents[:10]

    print(
        f"Using {len(historic_news)} "
        f"historic incidents"
    )

    final_data = []

    # =================================================
    # PROCESS
    # =================================================

    for idx, item in enumerate(historic_news):

        try:

            lat = item["coordinates"]["lat"]

            lon = item["coordinates"]["lon"]

            bbox = make_bbox(
                lat,
                lon
            )

            incident_date = datetime.strptime(
                item["date"],
                "%Y-%m-%d"
            )

            # ============================================
            # T1/T2
            # ============================================

            t1 = (
                incident_date -
                timedelta(days=10)
            )

            t2 = incident_date

            # ============================================
            # FETCH BEFORE IMAGE
            # ============================================

            before_img = fetch_image(

                bbox,

                (
                    t1.strftime("%Y-%m-%d"),
                    incident_date.strftime("%Y-%m-%d")
                )
            )

            # ============================================
            # FETCH AFTER IMAGE
            # ============================================

            after_img = fetch_image(

                bbox,

                (
                    t1.strftime("%Y-%m-%d"),
                    incident_date.strftime("%Y-%m-%d")
                )
            )

            # ============================================
            # IMAGE PATHS
            # ============================================

            before_name = (
                f"{idx}_before.png"
            )

            after_name = (
                f"{idx}_after.png"
            )

            before_path_abs = os.path.join(
                IMAGE_DIR,
                before_name
            )

            after_path_abs = os.path.join(
                IMAGE_DIR,
                after_name
            )

            # ============================================
            # SAVE BEFORE IMAGE
            # ============================================

            if before_img is not None:

                plt.imsave(
                    before_path_abs,
                    before_img
                )

            else:

                create_placeholder(
                    before_path_abs
                )

            # ============================================
            # SAVE AFTER IMAGE
            # ============================================

            if after_img is not None:

                plt.imsave(
                    after_path_abs,
                    after_img
                )

            else:

                create_placeholder(
                    after_path_abs
                )

            # ============================================
            # ADD METADATA
            # ============================================

            item["before_date"] = (
                t1.strftime("%Y-%m-%d")
            )

            item["after_date"] = (
                t2.strftime("%Y-%m-%d")
            )

            item["images"] = {

                "before_rgb":

                    f"/static/historic_images/{before_name}",

                "after_rgb":

                    f"/static/historic_images/{after_name}"
            }

            final_data.append(item)

            print(
                f"Processed: "
                f"{item['title']}"
            )

        except Exception as e:

            print(
                "Historic processing error:",
                e
            )

    # =================================================
    # SAVE JSON
    # =================================================

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_data,
            f,
            indent=2
        )

    print("historic_incidents.json saved")

    return final_data