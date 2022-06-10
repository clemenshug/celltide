import pandas as pd
import numpy as np
from magicgui.widgets import Container, ComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from skimage.measure import regionprops_table

from itertools import chain
from typing import Tuple, TYPE_CHECKING

try:
    import napari
except ImportError:
    pass

# Avoid circular import
# https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
if TYPE_CHECKING:
    from .tide import ContactProfiler


class CellTideVis:
    def __init__(self, profiler: "ContactProfiler"):
        self.profiler = profiler
        self.viewer = napari.Viewer()
        rps = regionprops_table(self.profiler.label_image, properties=["centroid"])
        self.centroids = np.stack([rps["centroid-0"], rps["centroid-1"]], axis=-1)
        self.edge_layer = None
        self.point_layer = None
        self.pairs = np.array(list(self.profiler.intersections.keys()))
        self.selected_pair = tuple(self.pairs[0])
        self.selected_channel = 0
        self.selected_radius = 3
        self.cell_label_layer = None

    def show_intensity_image(self) -> None:
        self.viewer.add_image(self.profiler.intensity_image)

    def show_label_image(self) -> None:
        self.viewer.add_labels(self.profiler.label_image)

    def _update_plot(self) -> None:
        if any(
            x is None
            for x in (self.selected_channel, self.selected_pair, self.selected_radius)
        ):
            return
        try:
            self.viewer.layers.remove(self.cell_label_layer)
        except Exception:
            pass
        cell_outlines = [self.profiler.contours[i] for i in self.selected_pair]
        self.cell_label_layer = self.viewer.add_shapes(
            cell_outlines,
            shape_type="polygon",
            face_color="transparent",
            edge_color="green",
            edge_width=1,
            properties={"cell_id": [i + 1 for i in self.selected_pair]},
            text={
                "text": "{cell_id}",
                "size": 12,
                "color": "green",
            },
            name="Cell labels",
        )
        self.viewer.layers.selection = [self.point_layer]
        self.mpl_widget.figure.clear()
        axs = self.mpl_widget.figure.subplots(2)
        cell_rings_radii = list(range(1 - self.selected_radius, self.selected_radius))
        cell_rings = self.profiler.radial_regionprops(
            cell_rings_radii,
            cells=self.selected_pair,
        )
        contact_profile = self.profiler.contact_profile(
            self.selected_pair, max_radius=self.selected_radius
        )
        for i in self.selected_pair:
            df = cell_rings[cell_rings["label"] == i + 1]
            axs[0].plot(
                df["expansion_radius"],
                df[f"intensity_mean-{self.selected_channel}"],
                label=f"cell {i + 1}",
            )
        axs[0].legend()
        axs[0].set_xlabel("Cell expansion radius")
        axs[1].plot(
            cell_rings_radii,
            contact_profile[:, self.selected_channel][::-1],
        )
        axs[1].set_xlabel(
            f"Position {self.selected_pair[0] + 1} ↔️ {self.selected_pair[1] + 1}"
        )
        self.mpl_widget.draw()

    def show_graph(self) -> None:
        vector_coords = np.stack(
            [self.centroids[self.pairs[:, i], :] for i in range(2)], axis=1
        )
        point_coords = np.mean(vector_coords, axis=1)
        vector_coords[:, 1, :] -= vector_coords[:, 0, :]
        self.edge_layer = self.viewer.add_vectors(
            vector_coords, name="Neighborhood graph"
        )
        # self.edge_layer.mode = "select"
        # self.edge_layer.editable = True
        self.point_layer = self.viewer.add_points(
            point_coords, size=1, name="Edge selection"
        )
        # self.point_layer.mode = "select"
        self.mpl_widget = FigureCanvas(Figure(figsize=(5, 3)))
        self.viewer.window.add_dock_widget(self.mpl_widget)

        def select_pair(event):
            if not self.point_layer.selected_data:
                return
            pairs_idx = next(iter(self.point_layer.selected_data))
            cell_pair = tuple(self.pairs[pairs_idx])
            if cell_pair == self.selected_pair:
                return
            self.selected_pair = cell_pair
            print(cell_pair)
            self._update_plot()

        self.point_layer.events.highlight.connect(select_pair)

        radius_menu = ComboBox(label="Max radius", value=3, choices=list(range(2, 8)))
        channel_menu = ComboBox(
            label="Channel",
            value=0,
            choices=list(range(self.profiler.intensity_image.shape[0])),
        )
        menu_widget = Container(widgets=[radius_menu, channel_menu])
        self.viewer.window.add_dock_widget(menu_widget)

        def radius_changed(event):
            self.selected_radius = int(radius_menu.value)
            self._update_plot()

        def channel_changed(event):
            self.selected_channel = int(channel_menu.value)
            self._update_plot()

        radius_menu.changed.connect(radius_changed)
        channel_menu.changed.connect(channel_changed)
        self._update_plot()

    def show_contact(self, cell_pair: Tuple[int, int], max_radius: int) -> None:
        self.profiler.contact_profile(cell_pair, max_radius)
        target_slice = self.profiler.profile_masks[max_radius][cell_pair][0]
        for img in chain(
            self.profiler.profile_masks[max_radius][cell_pair][1],
            chain.from_iterable(
                self.profiler.annuli[max_radius][cell_pair][1].values()
            ),
            [
                self.profiler.get_expanded_cell_mask(
                    i, max_radius, crop_slice=target_slice
                )[1]
                for i in cell_pair
            ],
        ):
            self.viewer.add_labels(
                img.astype(np.int32),
                translate=[sl.start for sl in target_slice],
                opacity=0.5,
            )
