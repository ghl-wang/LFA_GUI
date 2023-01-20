import os
import tkinter as tk
from PIL import Image, ImageTk
import pandas as pd
from tkinter import Menu, Label, Toplevel, Entry, filedialog, Button, simpledialog
import numpy as np
import matplotlib.pyplot as plt
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

    def __init__(self, canvas,root, peak_spacing):
        self.canvas = canvas
        self.parent = root
        self.canv_width = self.canvas.cget('width')
        self.canv_height = self.canvas.cget('height')
        self.original_image = None
        self.count = 0
        self.n = 3
        self.peak_spacing=peak_spacing
        self.reset()

        self.df = pd.DataFrame()

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
                                        parent=self.parent, initialvalue=self.count)
        if self.sample_label:
            self.crop_ROI()
            self.save_coordinates()
        self.hide()  # Hide cross-hairs.
        self.reset()

    def crop_ROI(self):
        left, top = [self.canvas.aspect*i for i in self.start]
        right, bottom = [self.canvas.aspect*i for i in self.end]
        roi = self.original_image.crop((left,top,right,bottom))
        roi_gray = roi.convert('L')
        nleft, nright = self.calculate_LR_border(roi_gray)
        roi_tight_gray = roi_gray.crop((nleft, 0, nright,roi.size[1]))
        roi_tight_color = roi.crop((nleft, 0, nright,roi.size[1]))
        line_peaks = [self.find_lfa_peaks(cropped_roi) for cropped_roi in [roi_tight_gray]+list(roi_tight_color.split())]
        
        file_title = self.file_label+'-'+str(self.count)+'-'+self.sample_label
        save_path = os.path.normpath(os.path.join(self.dir_name, self.file_label, file_title+'.png'))
        #roi_tight.save(save_path)

        n_channels = len(line_peaks)
        n_peaks = len(line_peaks[0][1])-1
        color_channels = ['gray', 'red', 'green', 'blue']
        features = [f'peak {i}' for i in range(n_peaks)] + ['background']
        data_types = ['index', 'signal']
        
        fig, axes = plt.subplots(nrows=1, ncols=1+n_channels, sharex=False, sharey=True)
        
        fig.suptitle(file_title)
        axes[0].imshow(roi_tight_color, aspect='auto')
        axes[0].set_xticks([])
        axes[0].set_ylabel('distance (pixels)')
        for i in range(0, n_channels):
            axes[i+1].set_xlabel(f'{color_channels[i]} (signal)')
            axes[i+1].plot(line_peaks[i][0], range(0,len(line_peaks[i][0])), color_channels[i])
            axes[i+1].plot(line_peaks[i][2][:-1], line_peaks[i][1][:-1], 'o', color=color_channels[i])
            axes[i+1].set_xlim([-25,255])
            axes[i+1].grid(True, which='major', color='lightgray')
            axes[i+1].set_xticks([0,100,200])
        fig.subplots_adjust(hspace=0, wspace=0)
        # for ax in axes:
        #      ax.set_aspect(2, share=True)
        fig.savefig(save_path)
        plt.close(fig)

        # n+1 to accommodate background intensity value appended to top 3 peaks
        # peak_labels = [f'peak {i}' for i in range(1,self.n+2)]
        # peak_labels[3] = 'background'
        # df = pd.DataFrame({'sample label': [self.sample_label]*(self.n+1),
        #                    'feature': peak_labels
        #                   })
        # for i, line_peak in enumerate(line_peaks):
        #     df[f'{color_channels[i]} index'] = line_peak[1]
        #     df[f'{color_channels[i]} signal'] = line_peak[2]

        # self.df=pd.concat([self.df, df])
        data=[[' '.join([c,f,t]), line_peaks[i][k+1][j]] for i,c in enumerate(color_channels) for j,f in enumerate(features) for k,t in enumerate(data_types) if f+t!='backgroundindex']
        df = pd.DataFrame([self.count]+[row[1] for row in data], index=['selection']+[row[0] for row in data], columns=[self.sample_label])
        self.df=pd.concat([self.df, df], axis=1)

    def calculate_LR_border(self, image):
        arr=np.asarray(image)
        mean_vertical=np.mean(arr,axis=0)
        gradient=np.gradient(mean_vertical)
        halfpoint=int(gradient.size//2)
        left=np.argmin(gradient[:halfpoint])
        right=np.argmin(gradient[halfpoint:])+halfpoint

        #new_crop_image=image.crop((left,0,right,image.size[1]))
        # new_crop_image.save('3.png')

        return left, right

    def find_lfa_peaks(self, cropped_image):

        arr = np.asarray(cropped_image)
        mean_horizontal=255-np.mean(arr,axis=1)
        filtered=savgol_filter(mean_horizontal, 13, 2)
        # switch to returning peaks > 3*sd above background (= 50 lowest values)?
        lowest_length = np.clip(len(filtered)//2, 1, 50)-1
        lowest = np.sort(filtered)[0:lowest_length]
        background = np.mean(lowest) #+ 3*np.std(lowest)
        peaks,_=find_peaks(filtered,distance=self.peak_spacing)
        # peaks,_=find_peaks(filtered, threshold=3*np.std(lowest))
        peak_height=filtered[peaks]
        peak_index_sorted=np.argsort(peak_height)
        peak_loc_sorted=peaks[peak_index_sorted]
        peak_height_sorted=peak_height[peak_index_sorted]

        # # top self.n peaks
        peak_loc_top3=peak_loc_sorted[-self.n:]
        peak_height_top3=peak_height_sorted[-self.n:]

        while len(peak_loc_top3) < 3:
            peak_loc_top3 = np.append(peak_loc_top3, 0)
        while len(peak_height_top3) < 3:
            peak_height_top3 = np.append(peak_height_top3, 0)

        # # sort by peak location
        peak_index_by_location=np.argsort(peak_loc_top3)
        peak_sort_by_location=np.append(peak_loc_top3[peak_index_by_location], 0)
        peak_height_sorted_by_location=np.append(peak_height_top3[peak_index_by_location], background)
        # peak_index_by_location=np.argsort(peak_loc_sorted)
        # peak_sort_by_location=peak_loc_sorted[peak_index_by_location]
        # peak_height_sorted_by_location=peak_loc_sorted[peak_index_by_location]

        return filtered, peak_sort_by_location, peak_height_sorted_by_location

    def save_coordinates(self):
        self.df.to_csv(self.csv_save_path, index=True)

    def update_data(self, image, filename):
        self.count = 0
        self.original_image = image
        (self.dir_name, self.file_name) = os.path.split(filename)
        (self.file_label, self.file_ext) = os.path.splitext(self.file_name)
        folder = os.path.normpath(os.path.join(self.dir_name, self.file_label))
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.csv_save_path = os.path.normpath(os.path.join(self.dir_name, self.file_label, self.file_label+'.csv'))
        #self.save_folder=folder

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

class popupWindow(object):
    def __init__(self,master):
        top=self.top=Toplevel(master)
        self.l=Label(top,text="Please input peak spacing value")
        self.l.pack()
        self.e=Entry(top)
        self.e.pack(pady=10, padx=20)
        self.b=Button(top,text='Ok',command=self.cleanup)
        self.b.pack()
    def cleanup(self):
        self.value=int(self.e.get())
        self.top.destroy()

class Application(tk.Frame):

    # Default selection object options.
    SELECT_OPTS = dict(dash=(2, 2), stipple='gray25', fill='red',
                          outline='')

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent=parent
        self.create_menu()

        path = "./front.png"
        # modify code to make image adjusted to window size
        bgimg = Image.open(path)
        self.img = ImageTk.PhotoImage(bgimg)
        self.canvas = tk.Canvas(root, width=self.img.width(), height=self.img.height(),
                                borderwidth=0, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.img_container=self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)
        self.canvas.img = self.img  # Keep reference.
        self.canvas.aspect = 1

        # Create selection object to show current selection boundaries.
        self.selection_obj = SelectionObject(self.canvas, self.SELECT_OPTS)
        self.peak_spacing=80
        # Callback function to update it given two points of its diagonal.


        # Create mouse position tracker that uses the function.
        self.posn_tracker = MousePositionTracker(self.canvas,parent, self.peak_spacing)
        self.posn_tracker.autodraw(command=self.on_drag)  # Enable callbacks.
        # self.button=Button(root, text='Save')
        #
        # self.button.pack(expand=True)
        #

    def on_drag(self, start, end, **kwarg):  # Must accept these arguments.
        self.selection_obj.update(start, end)

    def create_menu(self):
        self.menu_bar = Menu(self.parent)
        self.file_menu = Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(
            label="Open...", command=self.open_file)
        # add options to adjust peak spacings
        self.file_menu.add_command(
            label="Settings", command=self.adjust_settings)

        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.analysis_menu = Menu(self.menu_bar, tearoff=0)
        self.analysis_menu.add_command(
            label="Auto-analysis", command=self.auto_analysis)
        self.menu_bar.add_cascade(label="Analysis", menu=self.analysis_menu)

        self.parent.config(menu=self.menu_bar)

    def open_file(self, event=None):
        input_file_name = filedialog.askopenfilename(defaultextension=".txt",
                                                             filetypes=[("Image files", "*.png"), ("Image files", "*.tif"), ("All Files", "*.*")])
        if input_file_name:
            global file_name
            self.file_name = input_file_name
            root.title(f'{os.path.basename(self.file_name)}')
            img = Image.open(self.file_name)
            img_w, img_h = img.size
            canvas_w = self.canvas.winfo_width()
            self.canvas.aspect = img_w / canvas_w
            resized_img = img.resize((int(img_w / self.canvas.aspect), int(img_h / self.canvas.aspect)), Image.ANTIALIAS)
            self.original_img=img
            self.img = ImageTk.PhotoImage(resized_img)
            self.canvas.itemconfig(self.img_container, image=self.img)
            self.posn_tracker.update_data(self.original_img, self.file_name)

    def adjust_settings(self):
        self.settings_window=popupWindow(self.parent)
        self.parent.wait_window(self.settings_window.top)
        self.peak_spacing=self.settings_window.value
        print(self.peak_spacing)
        # Create mouse position tracker that uses the function.
        self.posn_tracker = MousePositionTracker(self.canvas,self.parent, self.peak_spacing)
        self.posn_tracker.autodraw(command=self.on_drag)  # Enable callbacks.
        self.posn_tracker.update_data(self.original_img, self.file_name)

    def auto_analysis(self, event=None):
        if self.posn_tracker.original_image != None:
            y_start = int(290/self.canvas.aspect)
            y_end = int(430/self.canvas.aspect)
            x_start = int(86/self.canvas.aspect)
            x_end = int(self.posn_tracker.original_image.size[0]/self.canvas.aspect)
            spacing = int(87/self.canvas.aspect)
            x_list = [pos for pos in range(x_start, x_end, spacing)]
            if x_list[-1] != x_end:
                x_list = x_list+[x_end]
            for x1, x2 in zip(x_list[:-1], x_list[1:]):
                self.posn_tracker.start = (x1, y_start)
                self.posn_tracker.end = (x2, y_end)
                rectangle = self.canvas.create_rectangle(x1, y_start, x2, y_end)
                self.posn_tracker.quit(None)
                self.canvas.delete(rectangle)

if __name__ == '__main__':

    WIDTH, HEIGHT = 1500,750 #1568, 882
    BACKGROUND = 'grey'
    TITLE = 'Image Cropper'


    root = tk.Tk()
    root.title(TITLE)
    root.geometry('%sx%s' % (WIDTH, HEIGHT))
    root.configure(background=BACKGROUND)


    app = Application(root, background=BACKGROUND)
    app.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.TRUE)
    app.mainloop()