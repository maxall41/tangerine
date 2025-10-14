from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QDialog,
    QGraphicsItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
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


class PolygonROI(QGraphicsPolygonItem):
    def __init__(self, points):
        super().__init__(QPolygonF([QPointF(x, y) for x, y in points]))
        self.pts = points
        self.setPen(QPen(QColor(0, 255, 0), 2))
        self.setBrush(QColor(0, 255, 0, 40))
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.updating = False
        self.handles = []
        self.build_handles()

    def paint(self, painter, option, widget=None):
        # Remove the selection rectangle by clearing the selection state
        option.state &= ~QStyle.StateFlag.State_Selected
        super().paint(painter, option, widget)

    def build_handles(self):
        for h in self.handles:
            h.setParentItem(None)
            if self.scene():
                self.scene().removeItem(h)
        self.handles = []
        poly = self.polygon()
        for i in range(len(poly)):
            p = poly.at(i)
            h = VertexHandle(self, i, p, size=6)
            self.handles.append(h)

    def refresh_handles(self):
        poly = self.polygon()
        self.updating = True
        for i, h in enumerate(self.handles):
            if i < len(poly):
                h.index = i
                h.setPos(poly.at(i))
        self.updating = False

    def update_vertex(self, index, pos):
        self.updating = True
        poly = self.polygon()
        if 0 <= index < len(poly):
            poly[index] = QPointF(pos.x(), pos.y())
            self.setPolygon(poly)
        self.updating = False
        self.refresh_handles()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            if value:  # Selected
                self.setPen(QPen(QColor(255, 0, 0), 2))
                self.setBrush(QColor(255, 0, 0, 40))

                # --- NEW CODE: print and pickle the selected polygon ---
                poly = self.polygon()
                pts = [(poly.at(i).x(), poly.at(i).y()) for i in range(len(poly))]
                print("\nSelected polygon points:")
                print(pts)
                # -------------------------------------------------------
            else:  # Deselected
                self.setPen(QPen(QColor(0, 255, 0), 2))
                self.setBrush(QColor(0, 255, 0, 40))
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


class VertexHandle(QGraphicsRectItem):
    def __init__(self, roi, index, pos, size=6):
        super().__init__(-size / 2, -size / 2, size, size)
        self.roi = roi
        self.index = index
        self.setBrush(QColor(255, 255, 255))
        self.setPen(QPen(QColor(0, 0, 0), 1))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges,
        )
        self.setZValue(10)
        self.setParentItem(roi)
        self.setPos(pos)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if not self.roi.updating:
                self.roi.update_vertex(self.index, value)
        return super().itemChange(change, value)
