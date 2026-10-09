"""Explicit controller fields, linked plots, and a shared inspection cursor."""

import json

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

ROLE = Qt.ItemDataRole.UserRole
CHECKED, UNCHECKED = Qt.CheckState.Checked, Qt.CheckState.Unchecked
COLOURS = (
    "#0072b2",
    "#d55e00",
    "#009e73",
    "#cc79a7",
    "#e69f00",
    "#56b4e9",
    "#6a3d9a",
    "#a65628",
    "#555555",
    "#bcbd22",
    "#17becf",
    "#e377c2",
    "#8c564b",
    "#9467bd",
    "#8c6d31",
    "#31a354",
    "#637939",
    "#6baed6",
    "#b15928",
    "#393b79",
)


def label(key):
    name, component = key
    return name if component is None else f"{name}[{component}]"


class PlotToolbar(NavigationToolbar2QT):
    toolitems = tuple(t for t in NavigationToolbar2QT.toolitems if t[0] in {"Pan", "Zoom"})


class Plot(QWidget):
    def __init__(self, owner, state=None):
        super().__init__()
        self.owner, self.fields, self.artists, self.signature = owner, [], {}, None
        self.drawing, self.manual, self.xy_samples = False, False, None
        self.figure = Figure(figsize=(6, 2.2), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(155)
        self.mode = QComboBox()
        self.mode.addItems(["Time series", "Heatmap", "X–Y", "Value"])
        self.swap = QPushButton("Swap X/Y")
        self.save_plot = QPushButton("Save plot…")
        self.toolbar = PlotToolbar(self.canvas, self, coordinates=False)
        self.message = QLabel()
        self.message.setWordWrap(True)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.mode)
        row.addWidget(self.swap)
        row.addWidget(self.toolbar)
        row.addStretch()
        row.addWidget(self.save_plot)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self.message)
        layout.addWidget(self.canvas, 1)
        self.mode.currentIndexChanged.connect(self.changed)
        self.swap.clicked.connect(self.swap_axes)
        self.save_plot.clicked.connect(self.export_plot)
        self.canvas.mpl_connect("button_press_event", self.pick)
        if isinstance(state, dict):
            for key in state.get("fields", [])[:4096]:
                if (
                    isinstance(key, list)
                    and len(key) == 2
                    and isinstance(key[0], str)
                    and (key[1] is None or type(key[1]) is int and 0 <= key[1] < 4096)
                ):
                    if tuple(key) not in self.fields:
                        self.fields.append(tuple(key))
            self.mode.setCurrentIndex(max(0, min(3, int(state.get("mode", 0)))))

    def settings(self):
        return {"fields": self.fields, "mode": self.mode.currentIndex()}

    def changed(self):
        self.signature, self.manual = None, False
        self.owner.redraw()
        self.owner.changed.emit()
        QTimer.singleShot(0, self.owner.reveal_plot)

    def swap_axes(self):
        self.fields.reverse()
        self.changed()

    def pick(self, event):
        if (
            event.inaxes is not getattr(self, "axes", None)
            or self.toolbar.mode
            or event.button != 1
        ):
            return
        if self.mode.currentIndex() == 2 and self.xy_samples is not None:
            times, x, y = self.xy_samples
            pixels = self.axes.transData.transform(np.column_stack((x, y)))
            distance = np.sum((pixels - [event.x, event.y]) ** 2, axis=1)
            valid = np.isfinite(distance)
            if valid.any():
                index = np.argmin(np.where(valid, distance, np.inf))
                self.owner.seekRequested.emit(float(times[index]))
        elif self.mode.currentIndex() in (0, 1) and event.xdata is not None:
            self.owner.seekRequested.emit(float(event.xdata))

    def limits_changed(self, axes):
        if self.drawing or not self.isVisible():
            return
        self.manual = True
        if self.mode.currentIndex() in (0, 1):
            self.owner.set_range(axes.get_xlim())

    def draw(self):
        self.drawing = True
        try:
            self._draw()
        finally:
            self.drawing = False

    def _draw(self):
        mode = self.mode.currentIndex()
        self.swap.setVisible(mode == 2)
        self.xy_samples = None
        keys = [key for key in self.fields if self.owner.available(key)]
        signature = mode, tuple((key, np.shape(self.owner.current(key))) for key in keys)
        if signature != self.signature:
            self.figure.clear()
            self.axes = self.figure.add_subplot()
            self.axes.callbacks.connect("xlim_changed", self.limits_changed)
            self.axes.callbacks.connect("ylim_changed", self.limits_changed)
            self.artists = {}
            self.signature, self.manual = signature, False
        axes = self.axes
        message = ""
        series = {key: self.owner.series(key) for key in keys}
        if not keys:
            message = "Check fields in the table to add them to this plot."
        elif mode == 0:
            if any(series[key] is None for key in keys):
                message = "No time history for these fields. Use Value for a snapshot."
            else:
                for key, (times, values) in series.items():
                    if np.ndim(values) != 1:
                        message = "Select vector components for time series."
                        break
                    if key not in self.artists:
                        (self.artists[key],) = axes.plot(
                            [], [], label=label(key), color=self.owner.colour(key)
                        )
                    self.artists[key].set_data(times, values)
                if not message:
                    axes.relim()
                    axes.autoscale_view(scalex=False, scaley=not self.manual)
                    axes.set_xlabel("Simulated seconds")
                    if "legend" not in self.artists:
                        self.artists["legend"] = axes.legend(
                            fontsize=7, loc="upper left", ncols=min(3, len(keys))
                        )
        elif mode == 1:
            names = {key[0] for key in keys}
            if len(names) != 1 or any(
                key[1] is None or series[key] is None for key in keys
            ):
                message = "Select components of one vector for a heatmap."
            else:
                times = series[keys[0]][0]
                values = np.array([series[key][1] for key in keys])
                # Frozen batch rows can repeat a timestamp; keep their first sample.
                unique = np.r_[True, np.diff(times) > 0]
                times, values = times[unique], values[:, unique]
                left, right = self.owner.range()
                begin, end = np.searchsorted(times, (left, right))
                begin, end = max(0, begin - 1), min(len(times), end + 1)
                times, values = times[begin:end], values[:, begin:end]
                if not len(times):
                    message = "No samples in this time interval."
                else:
                    # Use actual timestamps. A pcolormesh preserves nonuniform spacing.
                    edges = np.r_[
                        times,
                        times[-1] + (times[-1] - times[-2] if len(times) > 1 else 0.02),
                    ]
                    mesh = self.artists.get("mesh")
                    if mesh is not None:
                        mesh.remove()
                    mesh = axes.pcolormesh(
                        edges,
                        np.arange(len(keys) + 1) - 0.5,
                        np.ma.masked_invalid(values),
                        shading="flat",
                        cmap="viridis",
                    )
                    self.artists["mesh"] = mesh
                    if "bar" not in self.artists:
                        self.artists["bar"] = self.figure.colorbar(mesh, ax=axes, pad=0.02)
                    else:
                        self.artists["bar"].update_normal(mesh)
                    axes.set_yticks(np.arange(len(keys)), [str(key[1]) for key in keys])
                    axes.set_ylabel(keys[0][0] + " · component")
                    axes.set_xlabel("Simulated seconds")
        elif mode == 2:
            if len(keys) != 2 or any(
                series[key] is None or np.ndim(series[key][1]) != 1 for key in keys
            ):
                message = "Choose exactly two scalars or vector components for X–Y."
            else:
                (tx, x), (ty, y) = (series[key] for key in keys)
                # Only pair observations at the same recorded controller call.
                times, ix, iy = np.intersect1d(tx, ty, return_indices=True)
                x, y = x[ix], y[iy]
                self.xy_samples = times, x, y
                if "xy" not in self.artists:
                    (self.artists["xy"],) = axes.plot(
                        [],
                        [],
                        color=self.owner.colour(keys[1]),
                        label=f"{label(keys[1])} vs {label(keys[0])}",
                    )
                    (self.artists["point"],) = axes.plot(
                        [], [], "o", color=self.owner.colour(keys[1])
                    )
                self.artists["xy"].set_data(x, y)
                self.artists["point"].set_data(x[-1:], y[-1:])
                axes.set_xlabel(label(keys[0]))
                axes.set_ylabel(label(keys[1]))
                if not self.manual:
                    axes.relim()
                    axes.autoscale_view()
                if "legend" not in self.artists:
                    self.artists["legend"] = axes.legend(fontsize=7)
        else:
            if len({key[0] for key in keys}) != 1:
                message = "Choose one field for its current value."
            else:
                value = np.asarray(self.owner.snapshot[keys[0][0]])
                if value.ndim == 1 and value.size:
                    if "vector" not in self.artists:
                        (self.artists["vector"],) = axes.plot([], [], "o-")
                    self.artists["vector"].set_data(np.arange(len(value)), value)
                    axes.relim()
                    axes.autoscale_view()
                    axes.set_xlabel("Component")
                    axes.set_title(keys[0][0], fontsize=9)
                elif value.ndim != 2 or not value.size:
                    message = "Current values are shown beside the field table. Use Time series for scalars."
                else:
                    if "image" not in self.artists:
                        self.artists["image"] = axes.imshow(
                            value, aspect="auto", interpolation="nearest"
                        )
                        self.figure.colorbar(self.artists["image"], ax=axes)
                    else:
                        self.artists["image"].set_data(value)
                        self.artists["image"].autoscale()
                    axes.set_title(label(keys[0]), fontsize=9)
        self.message.setText(message)
        self.message.setVisible(bool(message))
        self.canvas.setVisible(not message)
        self.toolbar.setEnabled(not message)
        self.save_plot.setEnabled(not message)
        if not message:
            if mode in (0, 1):
                axes.set_xlim(*self.owner.range())
                if "cursor" not in self.artists:
                    self.artists["cursor"] = axes.axvline(
                        self.owner.time, color="#555555", ls="--", lw=0.8
                    )
                self.artists["cursor"].set_xdata([self.owner.time, self.owner.time])
            axes.tick_params(labelsize=8)
            # Debug keys are plain user text, never Matplotlib math expressions.
            for text in (axes.title, axes.xaxis.label, axes.yaxis.label):
                text.set_parse_math(False)
            legend = axes.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_parse_math(False)
            self.canvas.draw_idle()

    def export_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save diagnostic plot", "diagnostic.png", "PNG (*.png);;SVG (*.svg)"
        )
        if path:
            try:
                self.figure.savefig(path)
            except Exception as exc:
                QMessageBox.warning(self, "Cannot save plot", str(exc))


class Diagnostics(QWidget):
    seekRequested = Signal(float)
    changed = Signal()

    def __init__(self, state=None, *, compact=False):
        super().__init__()
        self.snapshot, self.history, self.time = {}, None, 0.0
        self.plots, self.items, self.colours = [], {}, {}
        self.time_range, self.schema, self.updating = None, None, False
        self.table = QTreeWidget()
        self.table.setHeaderLabels(["Field", "●", "Value"])
        self.table.headerItem().setToolTip(1, "Curve colour / selected vector components")
        self.table.setObjectName("diagnostic-fields")
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumWidth(240)
        self.table.setColumnWidth(0, 130)
        self.table.setColumnWidth(1, 42)
        self.table.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setIndentation(14)
        self.table.itemChanged.connect(self.check_field)
        self.table.currentItemChanged.connect(self.inspect)
        self.value = QPlainTextEdit()
        self.value.setReadOnly(True)
        self.value.setMaximumHeight(55)
        self.empty = QLabel("No fields exported by debug() at this frame.")
        self.empty.setWordWrap(True)
        self.active = QComboBox()
        self.active.currentIndexChanged.connect(self.select_plot)
        self.add = QPushButton("+")
        self.add.setToolTip("Add a plot; field checkboxes edit the selected plot.")
        self.remove = QPushButton("−")
        self.remove.setToolTip("Remove the selected plot.")
        self.add.clicked.connect(lambda: self.add_plot())
        self.remove.clicked.connect(self.remove_plot)
        selection = QHBoxLayout()
        selection.addWidget(QLabel("Plot"))
        selection.addWidget(self.active, 1)
        selection.addWidget(self.add)
        selection.addWidget(self.remove)
        fields = QWidget()
        left = QVBoxLayout(fields)
        left.setContentsMargins(0, 0, 0, 0)
        left.addLayout(selection)
        left.addWidget(self.empty)
        left.addWidget(self.table, 1)
        left.addWidget(self.value)
        self.follow = QCheckBox("Follow time")
        self.follow.setChecked(True)
        self.window = QDoubleSpinBox()
        self.window.setRange(0, 3600)
        self.window.setSuffix(" s")
        self.window.setSpecialValueText("Whole run")
        self.window.setValue(10)
        self.window.setToolTip("Visible time window; zero shows the whole run.")
        self.whole = QPushButton("Whole run")
        self.whole.clicked.connect(self.show_all)
        self.follow.toggled.connect(self.time_settings)
        self.window.valueChanged.connect(self.time_settings)
        navigation = QHBoxLayout()
        navigation.addWidget(self.follow)
        navigation.addWidget(self.window)
        navigation.addWidget(self.whole)
        navigation.addStretch()
        self.plot_box = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_box)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.addStretch()
        self.scroll = scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.plot_box)
        scroll.verticalScrollBar().rangeChanged.connect(
            lambda *_: QTimer.singleShot(0, self.reveal_plot)
        )
        scroll.setMinimumWidth(270 if compact else 310)
        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(navigation)
        layout.addWidget(scroll, 1)
        split = QSplitter(Qt.Orientation.Vertical if compact else Qt.Orientation.Horizontal)
        split.addWidget(fields)
        split.addWidget(right)
        split.setSizes([270, 650])
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Diagnostics · check fields to plot; click a curve to inspect time")
        title.setWordWrap(True)
        outer.addWidget(title)
        outer.addWidget(split, 1)
        if compact:
            self.follow.hide()
            self.window.hide()
            self.whole.hide()
        state = state if isinstance(state, dict) else {}
        try:
            for key, colour in state.get("colours", {}).items():
                if (
                    isinstance(key, str)
                    and isinstance(colour, str)
                    and QColor(colour).isValid()
                ):
                    self.colours[key] = colour
            self.window.setValue(float(state.get("window", 10)))
            self.follow.setChecked(bool(state.get("follow", True)))
            limits = state.get("range")
            if (
                isinstance(limits, list)
                and len(limits) == 2
                and np.isfinite(limits).all()
                and limits[0] < limits[1]
            ):
                self.time_range = tuple(limits)
            for plot in state.get("plots", [{"mode": 3 if compact else 0}]):
                self.add_plot(plot)
        except (ValueError, TypeError, AttributeError):
            pass
        if not self.plots:
            self.add_plot()
        self.active.setCurrentIndex(0)
        self.show_snapshot({})

    def settings(self):
        return dict(
            plots=[p.settings() for p in self.plots],
            colours=self.colours,
            window=self.window.value(),
            follow=self.follow.isChecked(),
            range=self.time_range,
        )

    def add_plot(self, state=None):
        plot = Plot(self, state)
        self.plots.append(plot)
        self.plot_layout.insertWidget(len(self.plots) - 1, plot)
        self.active.addItem(f"{len(self.plots)}")
        self.active.setCurrentIndex(len(self.plots) - 1)
        self.remove.setEnabled(len(self.plots) > 1)
        self.redraw()
        self.changed.emit()
        return plot

    def remove_plot(self):
        if len(self.plots) <= 1:
            return
        index = self.active.currentIndex()
        plot = self.plots.pop(index)
        self.plot_layout.removeWidget(plot)
        plot.deleteLater()
        with QSignalBlocker(self.active):
            self.active.clear()
            self.active.addItems([str(i + 1) for i in range(len(self.plots))])
            self.active.setCurrentIndex(min(index, len(self.plots) - 1))
        self.remove.setEnabled(len(self.plots) > 1)
        self.update_checks()
        self.changed.emit()

    def select_plot(self, *args):
        self.update_checks()
        QTimer.singleShot(0, self.reveal_plot)

    def reveal_plot(self):
        if self.plots and self.active.currentIndex() >= 0:
            self.scroll.ensureWidgetVisible(self.plots[self.active.currentIndex()])

    def colour(self, key):
        identity = json.dumps(key)
        if identity not in self.colours:
            self.colours[identity] = COLOURS[len(self.colours) % len(COLOURS)]
        return self.colours[identity]

    def current(self, key):
        value = self.snapshot[key[0]]
        return value if key[1] is None else value[key[1]]

    def available(self, key):
        if key[0] not in self.snapshot:
            return False
        value = self.snapshot[key[0]]
        return (
            key[1] is None
            or isinstance(value, np.ndarray)
            and value.ndim == 1
            and key[1] < len(value)
        )

    def series(self, key):
        series = (
            self.history(key[0])
            if callable(self.history)
            else (self.history or {}).get(key[0])
        )
        if series is None:
            return None
        times, values = (np.asarray(v) for v in series)
        return times, values if key[1] is None else values[:, key[1]]

    def show_snapshot(self, snapshot, history=None, *, time=None):
        self.snapshot, self.history = snapshot, history
        self.time = float(time) if time is not None else self.time
        schema = tuple(
            (name, np.shape(value), isinstance(value, str))
            for name, value in snapshot.items()
        )
        if schema != self.schema:
            selected = self.table.currentItem()
            selected = tuple(selected.data(0, ROLE)) if selected else None
            expanded = {
                tuple(item.data(0, ROLE))
                for item in self.items.values()
                if item.isExpanded()
            }
            with QSignalBlocker(self.table):
                self.table.clear()
                self.items = {}
                for name, value in snapshot.items():
                    root = QTreeWidgetItem(self.table, [name, "", ""])
                    key = name, None
                    root.setData(0, ROLE, key)
                    self.items[key] = root
                    if not isinstance(value, str) and np.size(value):
                        root.setFlags(root.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        root.setCheckState(0, UNCHECKED)
                    else:
                        root.setFlags(root.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                    if isinstance(value, np.ndarray) and value.ndim == 1:
                        for i in range(len(value)):
                            child = QTreeWidgetItem(root, [f"[{i}]", "", ""])
                            child.setData(0, ROLE, (name, i))
                            child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                            child.setCheckState(0, UNCHECKED)
                            self.items[name, i] = child
                    root.setExpanded(key in expanded)
                if selected in self.items:
                    self.table.setCurrentItem(self.items[selected])
                elif self.table.topLevelItemCount():
                    self.table.setCurrentItem(self.table.topLevelItem(0))
            self.schema = schema
        with QSignalBlocker(self.table):
            for key, item in self.items.items():
                value = self.current(key)
                text = (
                    f"array {value.shape}"
                    if isinstance(value, np.ndarray)
                    else f"{value:.5g}"
                    if isinstance(value, float)
                    else str(value)
                )
                item.setText(2, text)
                item.setToolTip(0, label(key))
        self.empty.setVisible(not snapshot)
        self.table.setVisible(bool(snapshot))
        self.update_checks()
        self.inspect()
        self.redraw()

    def inspect(self, *args):
        item = self.table.currentItem()
        self.value.setVisible(item is not None)
        if item is not None:
            key = tuple(item.data(0, ROLE))
            value = self.current(key)
            self.value.setPlainText(f"{label(key)}: {value}")

    def check_field(self, item, column):
        if column != 0 or not self.plots:
            return
        plot = self.plots[self.active.currentIndex()]
        key = tuple(item.data(0, ROLE))
        keys = [tuple(item.child(i).data(0, ROLE)) for i in range(item.childCount())] or [
            key
        ]
        checked = item.checkState(0) == CHECKED
        for key in keys:
            if checked and key not in plot.fields:
                plot.fields.append(key)
                self.colour(key)
            elif not checked and key in plot.fields:
                plot.fields.remove(key)
        self.update_checks()
        plot.changed()

    def update_checks(self, *args):
        if not self.plots or self.active.currentIndex() < 0:
            return
        fields = self.plots[self.active.currentIndex()].fields
        with QSignalBlocker(self.table):
            for key, item in self.items.items():
                if item.childCount():
                    count = sum((key[0], i) in fields for i in range(item.childCount()))
                    item.setCheckState(
                        0,
                        CHECKED
                        if count == item.childCount()
                        else Qt.CheckState.PartiallyChecked
                        if count
                        else UNCHECKED,
                    )
                    item.setText(1, f"{count}/{item.childCount()}")
                elif item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    item.setCheckState(0, CHECKED if key in fields else UNCHECKED)
                    colour = self.colours.get(json.dumps(key))
                    item.setText(1, "■" if colour else "")
                    if colour:
                        item.setForeground(1, QColor(colour))

    def range(self):
        if not self.follow.isChecked() and self.time_range is not None:
            return self.time_range
        width = self.window.value()
        end = max(self.time, 0.02)
        return max(0.0, end - width) if width else 0.0, end

    def set_range(self, limits):
        self.time_range = tuple(float(v) for v in limits)
        with QSignalBlocker(self.follow):
            self.follow.setChecked(False)
        if not self.updating:
            self.redraw()
        self.changed.emit()

    def time_settings(self, *args):
        if self.follow.isChecked():
            self.time_range = None
        self.redraw()
        self.changed.emit()

    def show_all(self):
        with QSignalBlocker(self.window), QSignalBlocker(self.follow):
            self.window.setValue(0)
            self.follow.setChecked(True)
        for plot in self.plots:
            plot.manual = False
        self.time_settings()

    def redraw(self):
        if self.updating:
            return
        self.updating = True
        try:
            for plot in self.plots:
                plot.draw()
        finally:
            self.updating = False
