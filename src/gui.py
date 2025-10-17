from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QDialog,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from waitingspinnerwidget import QtWaitingSpinner


class GraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.drawing = False
        self.on_add_point = None
        self.on_finish_polygon = None
        self.on_move_pointer = None

    def setDrawingMode(self, enabled):
        self.drawing = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event):
        if event.buttons() != Qt.MouseButton.NoButton:
            return
        angle = event.angleDelta().y()
        if angle == 0:
            return
        factor = 1.1 if angle > 0 else 1 / 1.1
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if self.drawing:
            if event.button() == Qt.MouseButton.LeftButton:
                pos = self.mapToScene(event.position().toPoint())
                if self.on_add_point:
                    self.on_add_point(pos)
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.drawing:
            if self.on_finish_polygon:
                self.on_finish_polygon()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.on_move_pointer:
            pos = self.mapToScene(event.position().toPoint())
            self.on_move_pointer(pos)
        super().mouseMoveEvent(event)


class PolygonROI(QGraphicsPathItem):
    def __init__(self, pts, holes=None):
        super().__init__()
        self.pts = pts
        self.holes = holes if holes else []  # List of hole polygons (list of point lists)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        # Don't set ItemIsMovable - we don't want dragging
        self.setZValue(3)
        self.vertex_items = []
        self.hole_vertex_items = []  # Store vertex items for holes
        self.update_path()

    def update_path(self):
        """Create a QPainterPath with holes properly cut out."""
        path = QPainterPath()

        # Add the exterior ring
        if len(self.pts) >= 3:
            path.moveTo(QPointF(self.pts[0][0], self.pts[0][1]))
            for p in self.pts[1:]:
                path.lineTo(QPointF(p[0], p[1]))
            path.closeSubpath()

        # Subtract holes using Qt's path operations
        for hole_pts in self.holes:
            if len(hole_pts) >= 3:
                hole_path = QPainterPath()
                hole_path.moveTo(QPointF(hole_pts[0][0], hole_pts[0][1]))
                for p in hole_pts[1:]:
                    hole_path.lineTo(QPointF(p[0], p[1]))
                hole_path.closeSubpath()
                # Subtract the hole from the main path
                path = path.subtracted(hole_path)

        self.setPath(path)

        # Set styling based on selection state
        if self.isSelected():
            pen = QPen(QColor(255, 0, 0), 2)  # Red when selected
            brush = QColor(255, 0, 0, 50)
        else:
            pen = QPen(QColor(0, 255, 0), 2)  # Green normally
            brush = QColor(0, 255, 0, 50)

        self.setPen(pen)
        self.setBrush(brush)

    def boundingRect(self):
        """Override to prevent selection box from appearing."""
        return super().boundingRect()

    def shape(self):
        """Override to make selection area match the actual polygon shape."""
        return super().shape()

    def paint(self, painter, option, widget=None):
        """Override paint to remove the selection rectangle but keep highlighting."""
        from PyQt6.QtWidgets import QStyle

        # Remove the selection state from the option to prevent the dotted rectangle
        option.state &= ~QStyle.StateFlag.State_Selected
        super().paint(painter, option, widget)

    def itemChange(self, change, value):
        """Handle item changes, including selection changes."""
        from PyQt6.QtWidgets import QGraphicsItem

        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            # Update colors when selection state changes
            self.update_path()
            # Show/hide vertices based on selection
            if value:  # Selected
                self.show_vertices()
            else:  # Deselected
                self.hide_vertices()
        return super().itemChange(change, value)

    def show_vertices(self):
        """Show vertex editing handles for exterior and holes."""
        if self.vertex_items or self.hole_vertex_items:
            return  # Already showing

        # Create exterior vertices
        for i, p in enumerate(self.pts):
            v = VertexHandle(self, i, is_hole=False)
            v.setPos(QPointF(p[0], p[1]))
            if self.scene():
                self.scene().addItem(v)
            self.vertex_items.append(v)

        # Create hole vertices
        for hole_idx, hole_pts in enumerate(self.holes):
            hole_vertices = []
            for i, p in enumerate(hole_pts):
                v = VertexHandle(self, i, is_hole=True, hole_index=hole_idx)
                v.setPos(QPointF(p[0], p[1]))
                if self.scene():
                    self.scene().addItem(v)
                hole_vertices.append(v)
            self.hole_vertex_items.append(hole_vertices)

    def hide_vertices(self):
        """Hide vertex editing handles."""
        for v in self.vertex_items:
            if self.scene():
                self.scene().removeItem(v)
        self.vertex_items = []

        for hole_vertices in self.hole_vertex_items:
            for v in hole_vertices:
                if self.scene():
                    self.scene().removeItem(v)
        self.hole_vertex_items = []

    def update_vertex(self, idx, new_pos, is_hole=False, hole_index=None):
        """Update a vertex position."""
        local_x = new_pos.x()
        local_y = new_pos.y()

        if is_hole and hole_index is not None:
            self.holes[hole_index][idx] = [local_x, local_y]
        else:
            self.pts[idx] = [local_x, local_y]

        self.update_path()

    def polygon(self):
        """Return QPolygonF for compatibility (only exterior)."""
        return QPolygonF([QPointF(p[0], p[1]) for p in self.pts])


class VertexHandle(QGraphicsRectItem):
    """Handle for editing polygon vertices."""

    def __init__(self, parent_roi, vertex_index, is_hole=False, hole_index=None):
        r = 3
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.parent_roi = parent_roi
        self.vertex_index = vertex_index
        self.is_hole = is_hole
        self.hole_index = hole_index
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setBrush(QColor(255, 0, 0))
        self.setPen(QPen(QColor(255, 255, 255), 1))
        self.setZValue(10)

    def itemChange(self, change, value):
        from PyQt6.QtWidgets import QGraphicsItem

        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.parent_roi.update_vertex(self.vertex_index, value, self.is_hole, self.hole_index)
        return super().itemChange(change, value)


class ProgressDialog(QDialog):
    cancelled = pyqtSignal()

    def __init__(self, parent, title, text):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(400, 160)
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint,
        )

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        self.label = QLabel(text)
        layout.addWidget(self.label)

        self.spinner = QtWaitingSpinner(self)

        self.spinner.setRoundness(70.0)
        self.spinner.setMinimumTrailOpacity(15.0)
        self.spinner.setTrailFadePercentage(70.0)
        self.spinner.setNumberOfLines(12)
        self.spinner.setLineLength(10)
        self.spinner.setLineWidth(5)
        self.spinner.setInnerRadius(10)
        self.spinner.setRevolutionsPerSecond(5)
        self.spinner.setColor(QColor(51, 219, 18))

        self.spinner.start()
        layout.addWidget(self.spinner)

        layout.addStretch()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancelled.emit)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)
