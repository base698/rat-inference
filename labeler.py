#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox
import os
import shutil
import json
import random
from pathlib import Path
from PIL import Image, ImageTk


class RatLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Rat Image Labeler")
        self.root.geometry("1200x800")
        
        self.base_dir = Path("datasets/rat")
        self.unsorted_dir = self.base_dir / "unsorted"
        self.train_images_dir = self.base_dir / "images" / "train"
        self.val_images_dir = self.base_dir / "images" / "val"
        self.train_labels_dir = self.base_dir / "labels" / "train"
        self.val_labels_dir = self.base_dir / "labels" / "val"
        
        self.train_images_dir.mkdir(parents=True, exist_ok=True)
        self.val_images_dir.mkdir(parents=True, exist_ok=True)
        self.train_labels_dir.mkdir(parents=True, exist_ok=True)
        self.val_labels_dir.mkdir(parents=True, exist_ok=True)
        
        self.processed_file = self.base_dir / ".processed_images.json"
        self.load_processed_images()
        
        self.current_image = None
        self.current_image_path = None
        self.photo_image = None
        self.scale_factor = 1.0
        
        self.rect_start = None
        self.rect_id = None
        self.bounding_boxes = []
        self.current_boxes = []
        
        self.setup_ui()
        self.load_next_image()
    
    def load_processed_images(self):
        if self.processed_file.exists():
            with open(self.processed_file, 'r') as f:
                self.processed_images = set(json.load(f))
        else:
            self.processed_images = set()
    
    def save_processed_images(self):
        with open(self.processed_file, 'w') as f:
            json.dump(list(self.processed_images), f)
    
    def setup_ui(self):
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.status_label = ttk.Label(top_frame, text="Loading...", font=("Arial", 12))
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.remaining_label = ttk.Label(top_frame, text="", font=("Arial", 10))
        self.remaining_label.pack(side=tk.LEFT, padx=10)
        
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.canvas = tk.Canvas(canvas_frame, bg="gray", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Button(button_frame, text="Save to Train (70%)", 
                  command=lambda: self.save_image("train"), 
                  width=20).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Save to Val (30%)", 
                  command=lambda: self.save_image("val"), 
                  width=20).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Auto-Assign (70/30)", 
                  command=self.auto_assign, 
                  width=20).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Clear Boxes", 
                  command=self.clear_boxes, 
                  width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Discard/Skip", 
                  command=self.discard_image, 
                  width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Undo Last Box", 
                  command=self.undo_last_box, 
                  width=15).pack(side=tk.LEFT, padx=5)
        
        info_frame = ttk.Frame(self.root, padding="5")
        info_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.info_label = ttk.Label(info_frame, 
                                   text="Click and drag to draw bounding boxes around rats. " +
                                        "You can draw multiple boxes per image.", 
                                   font=("Arial", 10))
        self.info_label.pack()
        
        self.box_count_label = ttk.Label(info_frame, text="Boxes: 0", font=("Arial", 10))
        self.box_count_label.pack()
    
    def load_next_image(self):
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(self.unsorted_dir.glob(ext))
        
        unprocessed = [f for f in image_files if f.name not in self.processed_images]
        
        self.remaining_label.config(text=f"Remaining: {len(unprocessed)}/{len(image_files)}")
        
        if not unprocessed:
            self.status_label.config(text="No more images to process!")
            messagebox.showinfo("Complete", "All images have been processed!")
            return False
        
        self.current_image_path = unprocessed[0]
        self.status_label.config(text=f"Current: {self.current_image_path.name}")
        
        try:
            img = Image.open(self.current_image_path)
            
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            self.current_image = img
            
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width <= 1:
                canvas_width = 1000
            if canvas_height <= 1:
                canvas_height = 600
            
            scale_x = canvas_width / img.width
            scale_y = canvas_height / img.height
            self.scale_factor = min(scale_x, scale_y, 1.0)
            
            new_width = int(img.width * self.scale_factor)
            new_height = int(img.height * self.scale_factor)
            
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(img_resized)
            
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, 
                                    image=self.photo_image, anchor=tk.CENTER)
            
            self.canvas_offset_x = (canvas_width - new_width) // 2
            self.canvas_offset_y = (canvas_height - new_height) // 2
            
            self.bounding_boxes = []
            self.current_boxes = []
            self.update_box_count()
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            self.processed_images.add(self.current_image_path.name)
            self.save_processed_images()
            return self.load_next_image()
    
    def on_canvas_click(self, event):
        if self.current_image is None:
            return
        
        self.rect_start = (event.x, event.y)
        
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        
        self.rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="red", width=2
        )
    
    def on_canvas_drag(self, event):
        if self.rect_start and self.rect_id:
            x1, y1 = self.rect_start
            self.canvas.coords(self.rect_id, x1, y1, event.x, event.y)
    
    def on_canvas_release(self, event):
        if self.rect_start and self.current_image:
            x1, y1 = self.rect_start
            x2, y2 = event.x, event.y
            
            if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
                if self.rect_id:
                    self.canvas.delete(self.rect_id)
                self.rect_id = None
                self.rect_start = None
                return
            
            img_x1 = (min(x1, x2) - self.canvas_offset_x) / self.scale_factor
            img_y1 = (min(y1, y2) - self.canvas_offset_y) / self.scale_factor
            img_x2 = (max(x1, x2) - self.canvas_offset_x) / self.scale_factor
            img_y2 = (max(y1, y2) - self.canvas_offset_y) / self.scale_factor
            
            img_x1 = max(0, min(img_x1, self.current_image.width))
            img_y1 = max(0, min(img_y1, self.current_image.height))
            img_x2 = max(0, min(img_x2, self.current_image.width))
            img_y2 = max(0, min(img_y2, self.current_image.height))
            
            x_center = (img_x1 + img_x2) / 2 / self.current_image.width
            y_center = (img_y1 + img_y2) / 2 / self.current_image.height
            width = abs(img_x2 - img_x1) / self.current_image.width
            height = abs(img_y2 - img_y1) / self.current_image.height
            
            self.bounding_boxes.append((x_center, y_center, width, height))
            self.current_boxes.append(self.rect_id)
            
            self.canvas.itemconfig(self.rect_id, outline="green", width=2)
            
            self.rect_id = None
            self.rect_start = None
            self.update_box_count()
    
    def update_box_count(self):
        self.box_count_label.config(text=f"Boxes: {len(self.bounding_boxes)}")
    
    def clear_boxes(self):
        for box_id in self.current_boxes:
            self.canvas.delete(box_id)
        self.bounding_boxes = []
        self.current_boxes = []
        self.update_box_count()
    
    def undo_last_box(self):
        if self.current_boxes:
            last_box = self.current_boxes.pop()
            self.canvas.delete(last_box)
            self.bounding_boxes.pop()
            self.update_box_count()
    
    def save_image(self, split):
        if not self.current_image_path:
            return
        
        if not self.bounding_boxes:
            result = messagebox.askyesno("No Boxes", 
                                        "No bounding boxes drawn. Save anyway?")
            if not result:
                return
        
        if split == "train":
            image_dir = self.train_images_dir
            label_dir = self.train_labels_dir
        else:
            image_dir = self.val_images_dir
            label_dir = self.val_labels_dir
        
        base_name = self.current_image_path.stem
        ext = self.current_image_path.suffix
        
        counter = 0
        while True:
            if counter == 0:
                new_name = base_name
            else:
                new_name = f"{base_name}_{counter}"
            
            new_image_path = image_dir / f"{new_name}{ext}"
            new_label_path = label_dir / f"{new_name}.txt"
            
            if not new_image_path.exists():
                break
            counter += 1
        
        try:
            shutil.copy2(self.current_image_path, new_image_path)
            
            with open(new_label_path, 'w') as f:
                for box in self.bounding_boxes:
                    x_center, y_center, width, height = box
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            self.processed_images.add(self.current_image_path.name)
            self.save_processed_images()
            
            messagebox.showinfo("Success", 
                              f"Saved to {split}:\nImage: {new_image_path.name}\n" +
                              f"Label: {new_label_path.name}")
            
            self.load_next_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")
    
    def auto_assign(self):
        if random.random() < 0.7:
            self.save_image("train")
        else:
            self.save_image("val")
    
    def discard_image(self):
        if self.current_image_path:
            result = messagebox.askyesno("Discard", 
                                        "Mark this image as processed without saving?")
            if result:
                self.processed_images.add(self.current_image_path.name)
                self.save_processed_images()
                self.load_next_image()


def main():
    root = tk.Tk()
    app = RatLabeler(root)
    root.mainloop()


if __name__ == "__main__":
    main()