import traceback

from colors import OilRedOQuantificationResults

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    print("Starting tangerine...")
    print("This may take a few seconds")
    import os
    import pickle
    import sys
    from pathlib import Path

    import numpy as np
    from PIL import Image, ImageDraw
    from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QAction, QColor, QPainterPath, QPen, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QFileDialog,
        QGraphicsPathItem,
        QGraphicsPixmapItem,
        QGraphicsRectItem,
        QGraphicsScene,
        QMainWindow,
        QMessageBox,
        QToolBar,
    )
    from scipy.ndimage import label
    from skimage.measure import approximate_polygon, label
    from skimage.morphology import remove_small_holes, remove_small_objects

    from colors import quantify_oil_red_o_stain
    from gui import GraphicsView, PolygonROI, ProgressDialog
    from shapes import outlines_list, polygon_area
    from utils import auto_segment, to_qimage

    class SegmentationThread(QThread):
        finished = pyqtSignal(object)  # Emits the segmentation result
        error = pyqtSignal(str)  # Emits error message if something goes wrong

        def __init__(self, image_path):
            super().__init__()
            self.image_path = image_path

        def run(self):
            try:
                result = auto_segment(self.image_path)
                mask = result >= 0.5

                mask = remove_small_holes(mask, area_threshold=1000)
                mask = remove_small_objects(mask, min_size=2000)

                Image.fromarray((mask * 255).astype(np.uint8)).save("raw_mask.png")

                # Threshold the predictions.
                labeled = label(mask)

                # debug_img = label2rgb(labeled, bg_label=0)
                # Image.fromarray((debug_img * 255).astype(np.uint8)).save("debug_labeled_mask.png")
                # print("Debug labeled mask saved to: debug_labeled_mask.png")

                # Convert to ROIs.
                outlines = outlines_list(labeled)
                outlines = [o for o in outlines if o.shape[0]]
                self.finished.emit(outlines)
            except Exception as e:
                self.error.emit(str(e))
                print(traceback.format_exc())

    class OilRedOQuantificationThread(QThread):
        finished = pyqtSignal(object)  # Emits the segmentation result
        error = pyqtSignal(str)  # Emits error message if something goes wrong

        def __init__(self, image_array, uncropped_image=None):
            super().__init__()
            self.image_array = image_array
            self.uncropped_image = uncropped_image

        def run(self):
            try:
                oil_red_o_quantification_results = quantify_oil_red_o_stain(
                    self.image_array,
                    uncropped_image=self.uncropped_image,
                )
                self.finished.emit(oil_red_o_quantification_results)
            except Exception as e:
                print(f"OilRedOQuantificationThread error: {e}")
                print(traceback.format_exc())
                self.error.emit(str(e))

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Tangerine")
            self.view = GraphicsView()
            self.scene = QGraphicsScene()
            self.view.setScene(self.scene)
            self.setCentralWidget(self.view)
            self.image_array = None
            self.display_pixmap_item = None
            self.rois = []
            self.temp_points = []
            self.temp_path_item = None
            self.temp_vertex_items = []
            self.create_actions()
            self.update_actions()
            self.view.on_add_point = self.add_drawing_point
            self.view.on_finish_polygon = self.finish_drawing
            self.view.on_move_pointer = self.update_temp_path_cursor
            self.segmentation_thread = None
            self.oil_red_o_thread = None
            self.progress_dialog = None
            self.cancelled = False

        def create_actions(self):
            tb = QToolBar()
            self.addToolBar(tb)
            self.act_load = QAction("Load TIFF", self)
            self.act_load.triggered.connect(self.load_image)
            tb.addAction(self.act_load)
            self.act_segment = QAction("Segment", self)
            self.act_segment.triggered.connect(self.segment_and_fit)
            tb.addAction(self.act_segment)
            self.act_draw = QAction("Draw Polygon", self)
            self.act_draw.triggered.connect(self.toggle_draw_mode)
            tb.addAction(self.act_draw)

            tb.addSeparator()

            self.act_save = QAction("Save Cropped", self)
            self.act_save.triggered.connect(self.save_cropped)
            tb.addAction(self.act_save)
            self.act_quantify_oilred_o = QAction("Quantify Oil Red O", self)
            self.act_quantify_oilred_o.triggered.connect(self.quantify_oilred_o)
            tb.addAction(self.act_quantify_oilred_o)

        def quantify_oilred_o(self):
            cropped_image = self.get_cropped_image()

            if cropped_image is not None:
                print("Got cropped image for Oil Red O Quantification")
                self.oil_red_o_thread = OilRedOQuantificationThread(cropped_image, uncropped_image=self.image_array)
            else:
                self.oil_red_o_thread = OilRedOQuantificationThread(self.image_array)

            self.oil_red_o_thread.finished.connect(self.on_quantification_finished)
            self.oil_red_o_thread.error.connect(self.on_quantification_error)
            self.oil_red_o_thread.start()

            self.show_progress("Segmentation", "Performing automatic segmentation...")

            self.save_segmentation_masks()

        def on_quantification_finished(self, results: OilRedOQuantificationResults):
            self.hide_progress()
            if self.cancelled:
                self.cancelled = False
                return
            self.image_array = results.oil_red_o_cutout_image
            qimg = to_qimage(self.image_array if self.image_array.ndim in (2, 3) else self.image_array.squeeze())
            pix = QPixmap.fromImage(qimg)
            self.clear_scene()
            self.display_pixmap_item = QGraphicsPixmapItem(pix)
            self.scene.addItem(self.display_pixmap_item)
            self.scene.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.update_actions()

            result_string = f"Oil Red O area coverage: {results.oil_red_o_percent_coverage:.3}%\nOil Red O Droplet Count: {results.oil_red_o_droplet_count}\nOil Red O Mean Droplet Area: {results.oil_red_o_droplet_mean_area:.3}px"

            results_directory = Path("./results/")
            save_path = results_directory / Path(self.image_name + ".txt")
            os.makedirs("./results/", exist_ok=True)

            with open(save_path, "w") as file:
                file.write(result_string)

            QMessageBox.information(self, "Result", f"Saving results to {save_path}\n{result_string}")

        @property
        def image_name(self):
            return self.image_path.split("/")[-1].split(".")[0]

        def save_segmentation_masks(self):
            save_directory = Path("./segmentation_rois/")
            os.makedirs(save_directory, exist_ok=True)
            save_path = save_directory / Path(self.image_name + ".pkl")
            all_points = [roi.pts for roi in self.rois]
            with open(save_path, "wb") as f:
                pickle.dump(all_points, f)

        def on_quantification_error(self, error_msg):
            self.hide_progress()

            QMessageBox.critical(self, "Segmentation Error", f"Failed to segment image:\n{error_msg}")

        def update_actions(self):
            has_img = self.image_array is not None
            self.act_segment.setEnabled(has_img)
            self.act_draw.setEnabled(has_img)
            self.act_save.setEnabled(has_img and len(self.rois) > 0)
            self.act_quantify_oilred_o.setEnabled(has_img)

        def clear_scene(self):
            self.scene.clear()
            self.display_pixmap_item = None
            self.rois = []
            self.cancel_drawing()

        def load_image(self):
            path, _ = QFileDialog.getOpenFileName(self, "Open TIFF", "", "TIFF Images (*.tif *.tiff)")
            if not path:
                return
            try:
                img = Image.open(path)
                arr = np.array(img)

                white_mask = np.all(arr == 255, axis=-1)
                black_mask = np.all(arr == 0, axis=-1)
                replace_mask = white_mask | black_mask
                arr[replace_mask] = np.array([128, 128, 128])

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")
                return
            self.image_path = path
            self.image_array = arr
            self.clear_scene()
            qimg = to_qimage(arr if arr.ndim in (2, 3) else arr.squeeze())
            pix = QPixmap.fromImage(qimg)
            self.display_pixmap_item = QGraphicsPixmapItem(pix)
            self.scene.addItem(self.display_pixmap_item)
            self.scene.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.update_actions()

        def toggle_draw_mode(self):
            if self.image_array is None:
                return
            if not self.view.drawing:
                self.start_drawing()
            else:
                self.finish_drawing()

        def start_drawing(self):
            self.view.setDrawingMode(True)
            self.temp_points = []
            for it in self.temp_vertex_items:
                self.scene.removeItem(it)
            self.temp_vertex_items = []
            if self.temp_path_item:
                self.scene.removeItem(self.temp_path_item)
                self.temp_path_item = None
            path = QPainterPath()
            self.temp_path_item = QGraphicsPathItem(path)
            pen = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine)
            self.temp_path_item.setPen(pen)
            self.temp_path_item.setZValue(5)
            self.scene.addItem(self.temp_path_item)
            self.act_draw.setText("Finish Drawing")

        def cancel_drawing(self):
            self.view.setDrawingMode(False)
            self.temp_points = []
            if self.temp_path_item:
                self.scene.removeItem(self.temp_path_item)
                self.temp_path_item = None
            for it in self.temp_vertex_items:
                self.scene.removeItem(it)
            self.temp_vertex_items = []
            self.act_draw.setText("Draw Polygon")

        def add_drawing_point(self, pos):
            if self.image_array is None:
                return
            x = max(0, min(self.image_array.shape[1] - 1, pos.x()))
            y = max(0, min(self.image_array.shape[0] - 1, pos.y()))
            p = QPointF(x, y)
            self.temp_points.append(p)
            r = 2
            dot = QGraphicsRectItem(-r, -r, 2 * r, 2 * r)
            dot.setPos(p)
            dot.setBrush(QColor(255, 0, 0))
            dot.setPen(QPen(QColor(255, 0, 0)))
            dot.setZValue(6)
            self.scene.addItem(dot)
            self.temp_vertex_items.append(dot)
            self.update_temp_path()

        def update_temp_path_cursor(self, pos):
            if not self.temp_points:
                return
            self.update_temp_path(pos)

        def update_temp_path(self, cursor_pos=None):
            if not self.temp_path_item:
                return
            path = QPainterPath()
            if self.temp_points:
                path.moveTo(self.temp_points[0])
                for p in self.temp_points[1:]:
                    path.lineTo(p)
                if cursor_pos is not None:
                    path.lineTo(cursor_pos)
            self.temp_path_item.setPath(path)

        def finish_drawing(self):
            if not self.view.drawing:
                return
            if len(self.temp_points) >= 3:
                pts = [(p.x(), p.y()) for p in self.temp_points]
                roi = PolygonROI(pts)
                self.scene.addItem(roi)
                self.rois.append(roi)
            self.cancel_drawing()
            self.update_actions()

        def keyPressEvent(self, event):
            if event.key() == Qt.Key.Key_Escape and self.view.drawing:
                self.cancel_drawing()
                event.accept()
                return
            if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
                # Delete selected ROIs
                selected_items = self.scene.selectedItems()
                for item in selected_items:
                    if isinstance(item, PolygonROI) and item in self.rois:
                        self.scene.removeItem(item)
                        self.rois.remove(item)
                self.update_actions()
                event.accept()
                return
            super().keyPressEvent(event)

        def segment_and_fit(self):
            if self.image_array is None:
                return

            # Create custom progress dialog
            self.cancelled = False
            self.show_progress("Segmentation", "Performing automatic segmentation...")

            # Create and start segmentation thread
            self.segmentation_thread = SegmentationThread(self.image_path)
            self.segmentation_thread.finished.connect(self.on_segmentation_finished)
            self.segmentation_thread.error.connect(self.on_segmentation_error)
            self.segmentation_thread.start()

        def on_segmentation_finished(self, outlines):
            if self.cancelled:
                self.cancelled = False
                return

            self.hide_progress()

            for roi in list(self.rois):
                self.scene.removeItem(roi)
            self.rois = []

            for pts in outlines:
                simplified_pts = approximate_polygon(pts, tolerance=8.0)

                if len(simplified_pts) < 3:
                    continue

                # 2. Calculate the area of the current polygon.
                area = polygon_area(simplified_pts)

                # 3. Check if the area meets the threshold before adding it.
                if area >= 100:
                    roi = PolygonROI(simplified_pts.tolist())
                    self.scene.addItem(roi)
                    self.rois.append(roi)

            print("Finished region fitting!")
            self.save_segmentation_masks()
            self.update_actions()

        def on_segmentation_error(self, error_msg):
            if self.cancelled:
                return

            self.hide_progress()

            QMessageBox.critical(self, "Segmentation Error", f"Failed to segment image:\n{error_msg}")

        def show_progress(self, title, message):
            """Show a progress dialog with the given title and message."""
            if self.progress_dialog:
                self.progress_dialog.close()

            self.cancelled = False
            self.progress_dialog = ProgressDialog(self, title, message)
            self.progress_dialog.cancelled.connect(self._on_progress_cancelled)
            self.progress_dialog.show()

        def hide_progress(self):
            """Hide and cleanup the progress dialog."""
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None

        def _on_progress_cancelled(self):
            """Handle progress dialog cancellation."""
            self.cancelled = True
            self.hide_progress()

        def get_cropped_image(self):
            if self.image_array is None or len(self.rois) == 0:
                return None
            h = self.image_array.shape[0]
            w = self.image_array.shape[1]
            mask_img = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask_img)
            for roi in self.rois:
                poly = roi.polygon()
                pts = [(int(poly.at(i).x()), int(poly.at(i).y())) for i in range(len(poly))]
                draw.polygon(pts, fill=255)
            mask = np.array(mask_img, dtype=np.uint8)
            m = (mask > 0).astype(np.uint8)
            arr = self.image_array
            if arr.ndim == 2:
                out = (arr * m).astype(arr.dtype)
            else:
                out = (arr * m[..., None]).astype(arr.dtype)
            return out

        def save_cropped(self):
            out = self.get_cropped_image()
            if out is None:
                QMessageBox.critical(self, "Error", "Failed to crop image")
                return
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Cropped Image",
                "",
                "TIFF Images (*.tif *.tiff);;PNG Images (*.png)",
            )
            if not path:
                return
            try:
                Image.fromarray(out).save(path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image:\n{e}")

    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(QSize(1000, 700))
    w.show()
    sys.exit(app.exec())
