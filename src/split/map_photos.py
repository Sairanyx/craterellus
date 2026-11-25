"""Map of where each photo was taken: Eddie red, Benjamin blue, hollow = no mushroom."""
import folium
import pandas as pd

catalog = pd.read_csv("dataset/clean/catalog.csv")
photos = catalog[(catalog["type"] == "photo") & catalog["lat"].notna()]

colours = {"eddie": "#AEA39A", "benjamin": "#C98A2B"}

search = folium.Map(
    location=[photos["lat"].mean(), photos["lon"].mean()], # taking the mean of all the coordinates to set the default middle loading map position
    zoom_start=18,
    max_zoom=22,
    tiles=None,                       # basically no default background, we add our own below (the map)
)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    max_zoom=22,
    max_native_zoom=19,
).add_to(search)

for photo in photos.to_dict("records"):
    colour = colours.get(photo["source"], "gray")

    if photo["has_mushroom"] == 1:
        fill = colour
    else:
        fill = "black"

    folium.CircleMarker(
        location=[photo["lat"], photo["lon"]],
        radius=5,
        color=colour,
        fill=True,
        fill_color=fill,
        fill_opacity=0.8,
    ).add_to(search)

legend = """
<div style="position: fixed; top: 10px; right: 10px; z-index: 1000;
            background: gray; padding: 8px 12px; border-radius: 6px; font-size: 14px;">
    <b>Mushroom search<b><br>
    <span style="color: #5B4636;">●</span> Eddie<br>
    <span style="color: #C98A2B;">●</span> Benjamin<br>
  ○ no mushroom
</div>
"""
search.get_root().html.add_child(folium.Element(legend))

search.save("results/photo_map.html")
print("Saved results/photo_map.html")