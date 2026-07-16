#!/usr/bin/env python3
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
import os
from pathlib import Path
from PIL import Image, ImageTk


class DatasetCleaner:
    def __init__(self, root, base_dir="datasets/rat"):
        self.root = root
        self.root.title("Dataset Image Cleaner")
        self.root.geometry("1200x800")

        self.base_dir = Path(base_dir)
        self.train_images_dir = self.base_dir / "images" / "train"
        self.val_images_dir = self.base_dir / "images" / "val"
        self.train_labels_dir = self.base_dir / "labels" / "train"
        self.val_labels_dir = self.base_dir / "labels" / "val"

        self.current_image = None
        self.current_image_path = None
        self.photo_image = None
        self.scale_factor = 1.0
        self.current_split = None

        # Get all images from both train and val
        self.image_list = []
        self.load_image_list()
        self.current_index = 0

        self.setup_ui()
        self.load_current_image()

    def load_image_list(self):
        """Load all images from train and val directories"""
        self.image_list = []

        # Load from train
        if self.train_images_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in self.train_images_dir.glob(ext):
                    self.image_list.append(('train', img_path))

        # Load from val
        if self.val_images_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in self.val_images_dir.glob(ext):
                    self.image_list.append(('val', img_path))

        print(f"Found {len(self.image_list)} images to review")

    def setup_ui(self):
        # Top frame with status
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.status_label = ttk.Label(top_frame, text="Loading...", font=("Arial", 12))
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.progress_label = ttk.Label(top_frame, text="", font=("Arial", 10))
        self.progress_label.pack(side=tk.LEFT, padx=10)

        self.split_label = ttk.Label(top_frame, text="", font=("Arial", 10, "bold"))
        self.split_label.pack(side=tk.LEFT, padx=10)

        # Canvas for image display
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Button frame
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Large, prominent buttons
        remove_btn = tk.Button(button_frame, text="REMOVE",
                              command=self.remove_image,
                              bg="#ff4444", fg="white",
                              font=("Arial", 14, "bold"),
                              width=20, height=2)
        remove_btn.pack(side=tk.LEFT, padx=20, expand=True)

        keep_btn = tk.Button(button_frame, text="KEEP",
                            command=self.keep_image,
                            bg="#44ff44", fg="black",
                            font=("Arial", 14, "bold"),
                            width=20, height=2)
        keep_btn.pack(side=tk.LEFT, padx=20, expand=True)

        # Navigation frame
        nav_frame = ttk.Frame(self.root, padding="5")
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Button(nav_frame, text="← Previous",
                  command=self.previous_image,
                  width=15).pack(side=tk.LEFT, padx=10)

        ttk.Button(nav_frame, text="Next →",
                  command=self.next_image,
                  width=15).pack(side=tk.LEFT, padx=10)

        # Info label
        info_frame = ttk.Frame(self.root, padding="5")
        info_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.info_label = ttk.Label(info_frame,
                                   text="Review images: REMOVE to delete from dataset, KEEP to continue",
                                   font=("Arial", 10))
        self.info_label.pack()

        # Keyboard shortcuts
        self.root.bind('<Left>', lambda e: self.previous_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('r', lambda e: self.remove_image())
        self.root.bind('R', lambda e: self.remove_image())
        self.root.bind('k', lambda e: self.keep_image())
        self.root.bind('K', lambda e: self.keep_image())
        self.root.bind('<space>', lambda e: self.keep_image())

    def load_current_image(self):
        """Load the image at current_index"""
        if not self.image_list:
            self.status_label.config(text="No images in dataset!")
            messagebox.showinfo("Complete", "No images found in the dataset!")
            return False

        if self.current_index >= len(self.image_list):
            self.status_label.config(text="All images reviewed!")
            messagebox.showinfo("Complete", "You've reviewed all images in the dataset!")
            return False

        self.current_split, self.current_image_path = self.image_list[self.current_index]

        # Update labels
        self.status_label.config(text=f"Current: {self.current_image_path.name}")
        self.progress_label.config(text=f"Image {self.current_index + 1} / {len(self.image_list)}")
        self.split_label.config(text=f"[{self.current_split.upper()}]")

        try:
            img = Image.open(self.current_image_path)

            # Convert to RGB
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            elif img.mode == 'L':
                img = img.convert('RGB')

            self.current_image = img

            # Calculate scaling
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width <= 1:
                canvas_width = 1000
            if canvas_height <= 1:
                canvas_height = 600

            scale_x = canvas_width / img.width
            scale_y = canvas_height / img.height
            self.scale_factor = min(scale_x, scale_y, 1.0)

            new_width = max(1, int(img.width * self.scale_factor))
            new_height = max(1, int(img.height * self.scale_factor))

            # Resize image
            try:
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            except AttributeError:
                img_resized = img.resize((new_width, new_height), Image.LANCZOS)

            self.photo_image = ImageTk.PhotoImage(img_resized)

            # Display image
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2,
                                    image=self.photo_image, anchor=tk.CENTER)

            return True

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            # Skip to next image
            self.current_index += 1
            return self.load_current_image()

    def get_label_path(self, image_path, split):
        """Get the corresponding label file path for an image"""
        if split == "train":
            label_dir = self.train_labels_dir
        else:
            label_dir = self.val_labels_dir

        label_path = label_dir / f"{image_path.stem}.txt"
        return label_path

    def remove_image(self):
        """Remove the current image and its label from the dataset"""
        if not self.current_image_path:
            return

        result = messagebox.askyesno("Confirm Removal",
                                    f"Delete {self.current_image_path.name} from dataset?\n\n"
                                    f"This will remove both the image and its label file.")

        if not result:
            return

        try:
            # Delete image file
            if self.current_image_path.exists():
                self.current_image_path.unlink()
                print(f"Deleted image: {self.current_image_path}")

            # Delete corresponding label file
            label_path = self.get_label_path(self.current_image_path, self.current_split)
            if label_path.exists():
                label_path.unlink()
                print(f"Deleted label: {label_path}")

            # Remove from list
            self.image_list.pop(self.current_index)

            # Don't increment index since we removed current item
            # Load the image now at current index (which was the next image)
            if self.current_index >= len(self.image_list):
                self.current_index = max(0, len(self.image_list) - 1)

            self.load_current_image()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove image: {e}")

    def keep_image(self):
        """Keep the current image and move to next"""
        self.next_image()

    def next_image(self):
        """Move to next image"""
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_current_image()
        else:
            messagebox.showinfo("End", "This is the last image!")

    def previous_image(self):
        """Move to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
        else:
            messagebox.showinfo("Start", "This is the first image!")


def main():
    parser = argparse.ArgumentParser(description="Review and remove unwanted YOLO dataset images.")
    parser.add_argument(
        "--dataset-root",
        default="datasets/rat",
        help="Dataset root containing images/ and labels/ directories.",
    )
    args = parser.parse_args()

    root = tk.Tk()
    app = DatasetCleaner(root, base_dir=args.dataset_root)
    root.mainloop()


if __name__ == "__main__":
    main()
