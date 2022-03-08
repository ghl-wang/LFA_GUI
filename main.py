import os
import tkinter as tk
from PIL import Image, ImageTk
import pandas as pd
from tkinter import Tk, Menu, Label, Toplevel, Entry,filedialog, Button, simpledialog
import numpy as np
from scipy.signal import savgol_filter,find_peaks


class popupWindow(object):

    value_global=""
    def __init__(self,master):
        self.top=Toplevel(master)
        self.top.columnconfigure(0, weight=1)
        self.top.columnconfigure(1,weight=3)
        self.value=""
        
        self.l=Label(self.top,text="Sample Label")
        self.l.grid(column=0,row=0,sticky=tk.W, padx=5, pady=5)

        self.e=Entry(self.top)
        self.e.grid(column=1, row=0, sticky=tk.E, padx=5, pady=5)


        self.b=Button(self.top,text='Ok',command=self.cleanup)
        self.b.grid(column=1, row=3, sticky=tk.E, padx=5, pady=5)

    def cleanup(self):
        self.value=self.e.get()
        popupWindow.value_global=self.value
        self.top.destroy()


class MousePositionTracker(tk.Frame):
    """ Tkinter Canvas mouse position widget. """

    def __init__(self, canvas,root):
        self.canvas = canvas
        self.parent=root
        self.canv_width = self.canvas.cget('width')
        self.canv_height = self.canvas.cget('height')
        self.orginal_image=None
        self.count=0
        self.reset()

        column_names = ["ID","Sample Label","Index_gray","Height_gray","Index_red","Height_red"]

        self.df = pd.DataFrame(columns=column_names)

        # Create canvas cross-hair lines.
        xhair_opts = dict(dash=(3, 2), fill='white', state=tk.HIDDEN)
        self.lines = (self.canvas.create_line(0, 0, 0, self.canv_height, **xhair_opts),
                      self.canvas.create_line(0, 0, self.canv_width,  0, **xhair_opts))

    def cur_selection(self):
        return (self.start, self.end)

    def begin(self, event):
        self.hide()
        self.start = (event.x, event.y)  # Remember position (no drawing).

    def update(self, event):
        self.end = (event.x, event.y)
        self._update(event)
        self._command(self.start, (event.x, event.y))  # User callback.

    def _update(self, event):
        # Update cross-hair lines.
        self.canvas.coords(self.lines[0], event.x, 0, event.x, self.canv_height)
        self.canvas.coords(self.lines[1], 0, event.y, self.canv_width, event.y)
        self.show()

    def reset(self):
        self.start = self.end = None

    def hide(self):
        self.canvas.itemconfigure(self.lines[0], state=tk.HIDDEN)
        self.canvas.itemconfigure(self.lines[1], state=tk.HIDDEN)

    def show(self):
        self.canvas.itemconfigure(self.lines[0], state=tk.NORMAL)
        self.canvas.itemconfigure(self.lines[1], state=tk.NORMAL)

    def autodraw(self, command=lambda *args: None):
        """Setup automatic drawing; supports command option"""
        self.reset()
        self._command = command
        self.canvas.bind("<Button-1>", self.begin)
        self.canvas.bind("<B1-Motion>", self.update)
        self.canvas.bind("<ButtonRelease-1>", self.quit)

    def quit(self, event):
        self.count+=1
        self.sample_label = simpledialog.askstring("Input", "Sample label",
                                        parent=self.parent)
        if self.sample_label:
            self.crop_ROI()
            self.save_coordinates()
        self.hide()  # Hide cross-hairs.
        self.reset()

    def crop_ROI(self):
        left, top=[2*i for i in self.start]
        right, bottom=[2*i for i in self.end]
        roi=self.orginal_image.crop((left,top,right,bottom))
        roi_gray=roi.convert('L')
        red, green, blue = roi.split()
        nleft, nright,new_cropped_roi_gray=self.calculate_LR_border(roi_gray)
        _,_, new_cropped_roi_red = self.calculate_LR_border(red)

        save_path=os.path.join(self.save_folder,str(self.count)+'.png')
        roi_tight=roi.crop((nleft,0,nright,roi.size[1]))
        roi_tight.save(save_path)


        # new_left,new_right=self.calculate_LR_border(roi_gray)
        peak_loc_gray, peak_height_gray=self.find_lfa_peaks(new_cropped_roi_gray)
        peak_loc_red, peak_height_red=self.find_lfa_peaks(new_cropped_roi_red)

        self.df=pd.concat([self.df, pd.DataFrame({'ID': self.count,
                                                  "Sample Label": self.sample_label,
                                                  'Index_gray': peak_loc_gray,
                                                  'Height_gray': peak_height_gray,
                                                  'Index_red': peak_loc_red,
                                                  'Height_red': peak_height_red
                                                  })])

    def calculate_LR_border(self, image):
        arr=np.asarray(image)
        mean_vertical=np.mean(arr,axis=0)
        gradient=np.gradient(mean_vertical)
        halfpoint=gradient.size//2
        left=np.argmin(gradient[:halfpoint])
        right=np.argmin(gradient[halfpoint:])+halfpoint

        new_crop_image=image.crop((left,0,right,image.size[1]))
        # new_crop_image.save('3.png')

        return left, right, new_crop_image

    def find_lfa_peaks(self,new_cropped_image):

        arr = np.asarray(new_cropped_image)
        mean_horizontal=255-np.mean(arr,axis=1)
        filtered=savgol_filter(mean_horizontal, 13, 2)
        peaks,_=find_peaks(filtered)
        peak_height=filtered[peaks]
        peak_index_sorted=np.argsort(peak_height)
        peak_loc_sorted=peaks[peak_index_sorted]
        peak_height_sorted=peak_height[peak_index_sorted]

        # top 3 peaks
        peak_loc_top3=peak_loc_sorted[-3:]
        peak_height_top3=peak_height_sorted[-3:]

        # sort by peak location
        peak_index_by_location=np.argsort(peak_loc_top3)
        peak_sort_by_location=peak_loc_top3[peak_index_by_location]
        peak_height_sorted_by_location=peak_height_top3[peak_index_by_location]


        return peak_sort_by_location,peak_height_sorted_by_location

    def save_coordinates(self):
        save_path=os.path.join(self.save_folder,self.csv_filename)
        self.df.to_csv(save_path)

    def update_data(self, image, filename):
        self.count=0
        self.orginal_image=image
        splits =filename.split('/')
        png_file=splits[-1]
        self.csv_filename=png_file.replace(".png",".csv")
        folder=filename.replace(".png","")
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.save_folder=folder


class SelectionObject:
    """ Widget to display a rectangular area on given canvas defined by two points
        representing its diagonal.
    """
    def __init__(self, canvas, select_opts):
        # Create attributes needed to display selection.
        self.canvas = canvas
        self.select_opts1 = select_opts
        self.width = self.canvas.cget('width')
        self.height = self.canvas.cget('height')

        # Options for areas outside rectanglar selection.
        select_opts1 = self.select_opts1.copy()  # Avoid modifying passed argument.
        select_opts1.update(state=tk.HIDDEN)  # Hide initially.
        # Separate options for area inside rectanglar selection.
        select_opts2 = dict(dash=(2, 2), fill='', outline='white', state=tk.HIDDEN)

        # Initial extrema of inner and outer rectangles.
        imin_x, imin_y,  imax_x, imax_y = 0, 0,  1, 1
        omin_x, omin_y,  omax_x, omax_y = 0, 0,  self.width, self.height

        self.rects = (
            # Area *outside* selection (inner) rectangle.
            self.canvas.create_rectangle(omin_x, omin_y,  omax_x, imin_y, **select_opts1),
            self.canvas.create_rectangle(omin_x, imin_y,  imin_x, imax_y, **select_opts1),
            self.canvas.create_rectangle(imax_x, imin_y,  omax_x, imax_y, **select_opts1),
            self.canvas.create_rectangle(omin_x, imax_y,  omax_x, omax_y, **select_opts1),
            # Inner rectangle.
            self.canvas.create_rectangle(imin_x, imin_y,  imax_x, imax_y, **select_opts2)
        )

    def update(self, start, end):
        # Current extrema of inner and outer rectangles.
        imin_x, imin_y,  imax_x, imax_y = self._get_coords(start, end)
        omin_x, omin_y,  omax_x, omax_y = 0, 0,  self.width, self.height

        # Update coords of all rectangles based on these extrema.
        self.canvas.coords(self.rects[0], omin_x, omin_y,  omax_x, imin_y),
        self.canvas.coords(self.rects[1], omin_x, imin_y,  imin_x, imax_y),
        self.canvas.coords(self.rects[2], imax_x, imin_y,  omax_x, imax_y),
        self.canvas.coords(self.rects[3], omin_x, imax_y,  omax_x, omax_y),
        self.canvas.coords(self.rects[4], imin_x, imin_y,  imax_x, imax_y),

        for rect in self.rects:  # Make sure all are now visible.
            self.canvas.itemconfigure(rect, state=tk.NORMAL)

    def _get_coords(self, start, end):
        """ Determine coords of a polygon defined by the start and
            end points one of the diagonals of a rectangular area.
        """
        return (min((start[0], end[0])), min((start[1], end[1])),
                max((start[0], end[0])), max((start[1], end[1])))

    def hide(self):
        for rect in self.rects:
            self.canvas.itemconfigure(rect, state=tk.HIDDEN)


class Application(tk.Frame):

    # Default selection object options.
    SELECT_OPTS = dict(dash=(2, 2), stipple='gray25', fill='red',
                          outline='')

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent=parent
        self.create_file_menu()

        path = "front.png"
        # modify code to make image adjusted to window size
        bgimg = Image.open(path)

        self.img = ImageTk.PhotoImage(bgimg)
        self.canvas = tk.Canvas(root, width=self.img.width(), height=self.img.height(),
                                borderwidth=0, highlightthickness=0)
        self.canvas.pack(expand=True)

        self.img_container=self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)
        self.canvas.img = self.img  # Keep reference.

        # Create selection object to show current selection boundaries.
        self.selection_obj = SelectionObject(self.canvas, self.SELECT_OPTS)

        # Callback function to update it given two points of its diagonal.
        def on_drag(start, end, **kwarg):  # Must accept these arguments.
            self.selection_obj.update(start, end)

        # Create mouse position tracker that uses the function.
        self.posn_tracker = MousePositionTracker(self.canvas,parent)
        self.posn_tracker.autodraw(command=on_drag)  # Enable callbacks.
        # self.button=Button(root, text='Save')
        #
        # self.button.pack(expand=True)
        #
    def create_file_menu(self):
        self.menu_bar = Menu(self.parent)
        self.file_menu = Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(
            label="Open...", command=self.open_file)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.parent.config(menu=self.menu_bar)

    def open_file(self, event=None):
        input_file_name = filedialog.askopenfilename(defaultextension=".txt",
                                                             filetypes=[("Image files", "*.png"), ("All Files", "*.*")])
        if input_file_name:
            global file_name
            file_name = input_file_name
            root.title('{}'.format(os.path.basename(file_name)))
            bgimg = Image.open(input_file_name)
            width, height = bgimg.size
            resized_bgimg = bgimg.resize((width // 2, height // 2), Image.ANTIALIAS)

            self.img = ImageTk.PhotoImage(resized_bgimg)
            self.canvas.itemconfig(self.img_container, image=self.img)
            self.posn_tracker.update_data(bgimg, file_name)

if __name__ == '__main__':

    WIDTH, HEIGHT = 1225, 690
    BACKGROUND = 'grey'
    TITLE = 'Image Cropper'


    root = tk.Tk()
    root.title(TITLE)
    root.geometry('%sx%s' % (WIDTH, HEIGHT))
    root.configure(background=BACKGROUND)


    app = Application(root, background=BACKGROUND)
    app.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.TRUE)
    app.mainloop()