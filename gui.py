#! /usr/bin/som python
# mostly taken from Olivia Guest

import pygtk
pygtk.require('2.0')
import gtk, gobject, cairo, glib
from gtk import gdk
import gobject
import random as r
import copy as cop
import os

from minisom import MiniSom

import matplotlib
matplotlib.use('gtkagg')

from pylab import plot,axis,show,pcolor,colorbar,bone,ion, hold
import numpy as np


import matplotlib.pyplot as plt
#from numpy import arange, sin, pi

# uncomment to select /GTK/GTKAgg/GTKCairo
#from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
#from matplotlib.backends.backend_gtkcairo import FigureCanvasGTKCairo as FigureCanvas

# or NavigationToolbar for classic
#from matplotlib.backends.backend_gtk import NavigationToolbar2GTK as NavigationToolbar
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NavigationToolbar

# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler




def Random(max_value, min_value = 0):
  "Random integer from min_value to max_value"
  return int(r.randint(min_value, max_value))
  #return int(round(r.random()*(max_value-min_value) + min_value))



def div(l, d):
  return [x/d for x in l]

#red = [255, 20, 0]
#red = div(red, 255.0)
red = [1.0, 0.1, 0.2]
blue = [0.0, 0.1, 1.0]
#blue = div(blue,255.0)
#green = [50, 180, 30]
#green = div(green,255.0)
grey = [0.9, 0.9, 0.9]
#purple = [255, 20, 255]
#purple = div(purple,255.0)
#yellow = [255, 200, 10]
#yellow = div(yellow, 255.0)
#cyan = [0, 255, 255]
#cyan = div(cyan, 255.0)
white = [1, 1, 1]
black = [0, 0, 0]

def random_colour():
   colours = [red, blue]
   return colours[Random(len(colours)-1)]

def get_resource_path(rel_path):
  dir_of_py_file = os.path.dirname(__file__)
  rel_path_to_resource = os.path.join(dir_of_py_file, rel_path)
  abs_path_to_resource = os.path.abspath(rel_path_to_resource)
  return abs_path_to_resource
#Environment Globals

padding = 10

#height = 60
#width = 60

#ticks = 1

#Agent Globals
living_cost = 0.3
reproduction_energy = 10 #minmum energy value for reproducing
mutation_probability = 5 #per cent
mouth_size = 3 #amount of food that can be eaten in one go
s = 1
d = 0
max_age = 500
death_percent = 0#20
PERCEPTION = 2




#import pygame
#from pygame.locals import *

#pygame.init()
#screen = pygame.display.set_mode((569, 569))
#pygame.display.set_caption("Map")

#background = pygame.Surface(screen.get_size())
#background = background.convert()
#background.fill((255, 255, 255))
#screen.blit(background, (0, 0))


##----------------------------------------------------------------------------------------------------##

class SOM:



    def If_running(self):
      #print som.running
      self.play.set_sensitive(not self.som.running)
      return self.som.running

    def If_paused(self):
      #print som.running
      #self.pause.set_sensitive(self.som.running)
      return False

    def Status_update(self):
      if self.som.running:
	context_id = self.status_bar.get_context_id("Running")
	#print context_id
	text = "Iteration: " +  str(self.som.tick).zfill(len(str(self.som.ticks))) + "/" + str(self.som.ticks).zfill(len(str(self.som.ticks)))
	if self.som.paused:
	  text += ", Paused"
	self.status_bar.push(context_id, text)
	return True # we need it to keep updating if the model is running
      elif not self.som.running:
	if not self.som.paused:
	  self.status_bar.remove_all(self.status_bar.get_context_id("Running"))
	  self.status_bar.remove_all(self.status_bar.get_context_id("Ready"))
	  context_id = self.status_bar.get_context_id("Ready")
	  #print context_id
	  text = "Ready"
	  self.status_bar.push(context_id, text)
	return False

    #def Quit(self, widget, data=None):
      ##print 'Byez!'
      #gtk.main_quit()

    #def Pause(self, widget=None, data=None):
	#self.som.Pause()
	#if self.som.paused:
	  #self.pause.set_label("Unpause")
	#else:
	  #self.pause.set_label("Pause")
	  #glib.idle_add(self.som.Run)
	  #glib.idle_add(self.If_running)
	#glib.idle_add(self.Status_update)


    def open_file(self, file_name):
      try:
	  #cols = self.columns[self.combobox.get_active()]
	  #print cols
	  self.data = np.genfromtxt(file_name, delimiter=',',usecols=(self.visual_and_acoustic),skip_header=1)
	  self.pattern_labels = np.genfromtxt(file_name, delimiter=',',usecols=(self.visual_and_acoustic), skip_footer=14, dtype=str)
	  self.file_name = file_name

	  self.update_treeview(self.data, self.patterns_liststore)

	  #print self.data
      except:
	  print "File is probably not in the right format:", file_name
	  raise

    def select_file(self, widget=None, data=None):
      #response = self.dialog.run()
      #if response == gtk.RESPONSE_OK:
	#self.open_file(self.dialog.get_filename())

      #elif response == gtk.RESPONSE_CANCEL:
	#print 'Closed, no files selected'

      #self.dialog.destroy()

      dialog = gtk.FileChooserDialog("Open..", None, gtk.FILE_CHOOSER_ACTION_OPEN, (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_OPEN, gtk.RESPONSE_OK))
      dialog.set_default_response(gtk.RESPONSE_OK)
      tmp = os.getcwd()
      tmp = 'file://' + tmp
      #print tmp
      #print dialog.set_current_folder_uri(tmp)
      #print dialog.get_current_folder_uri()
      filter = gtk.FileFilter()
      filter.set_name("All files")
      filter.add_pattern("*")
      dialog.add_filter(filter)

      filter = gtk.FileFilter()
      filter.set_name("Comma-separated values")

      filter.add_pattern("*.csv")
      dialog.add_filter(filter)
      dialog.set_filter(filter)

        #dialog = gtk.FileChooserDialog("Please choose a file", self,
            #gtk.FileChooserAction.OPEN,
            #(gtk.STOCK_CANCEL, gtk.ResponseType.CANCEL,
             #gtk.STOCK_OPEN, gtk.ResponseType.OK))


      response = dialog.run()
      if response == gtk.RESPONSE_OK:
	  #print("Open clicked")
	  #print("File selected: " + dialog.get_filename())
	  self.open_file(dialog.get_filename())
      #elif response == gtk.RESPONSE_CANCEL:
	  #print("Cancel clicked")

      dialog.destroy()

    def Run(self, widget=None, data=None):
      #self.som.ticks += self.iterations_spin_button.get_value_as_int()

      if not self.som.running:
	### Initialization and training ###
	#self.som = MiniSom(5, 15, 8,sigma=1.2,learning_rate=0.5)
	#self.init_som()
	for i in range(1):
	  self.train_som()
	  #self.figure.clf()
	  self.Draw_figure()
	  self.canvas.draw()
	  self.canvas.draw_idle()
	  #We need to draw *and* flush
	  self.figure.canvas.draw()
	  self.figure.canvas.flush_events()
	  #print "draw"

	  self.update_treeview(self.test_data, self.test_liststore)
	  self.update_treeview(self.data, self.patterns_liststore)



	  glib.idle_add(self.Status_update)
	  glib.idle_add(self.If_running)
	  glib.idle_add(self.If_paused)


    def Test(self, widget=None, data=None):
      #self.som.ticks += self.iterations_spin_button.get_value_as_int()

      if not self.som.running:
	### Initialization and training ###
	#self.som = MiniSom(5, 15, 8,sigma=1.2,learning_rate=0.5)
	self.test_som()
	#self.figure.clf()
	self.Draw_figure()
	self.canvas.draw()
	self.canvas.draw_idle()
	#We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
	#print "draw"

      glib.idle_add(self.Status_update)
      glib.idle_add(self.If_running)
      glib.idle_add(self.If_paused)


    def Reset(self, widget=None, data=None):
      self.init_som()
      self.Draw_figure()
      self.canvas.draw()
      self.canvas.draw_idle()
      #We need to draw *and* flush
      self.figure.canvas.draw()
      self.figure.canvas.flush_events()
      #print "draw"

      self.update_treeview(self.test_data, self.test_liststore)
      self.update_treeview(self.data, self.patterns_liststore)



      glib.idle_add(self.Status_update)
      glib.idle_add(self.If_running)
      glib.idle_add(self.If_paused)



    def delete_event(self, widget=None, event=None, data=None):
        # If you return FALSE in the "delete_event" signal handler,
        # GTK will emit the "destroy" signal. Returning TRUE means
        # you don't want the window to be destroyed.
        # This is useful for popping up 'are you sure you want to quit?'
        # type dialogs.
        #print "delete event occurred"

        # Change FALSE to TRUE and the main window will not be destroyed
        # with a "delete_event".
        return False

    #def on_key_event(self, event):
      #print('you pressed %s'%event.key)
      #key_press_handler(event, self.canvas, self.toolbar)

    def destroy(self, widget=None, data=None):
        #print "destroy signal occurred"
        gtk.main_quit()

    def Draw_figure(self):
  	self.axes.cla()   # Clear axis
	cols = self.columns[self.combobox.get_active()]
	data = self.data[:, 0:len(cols)]


	#ion()       # Turn on interactive mode.
	#hold(True) # Clear the plot before adding new data.


	#print som.distance_map().T
	#exit()
	bone()

	background = self.axes.pcolor(self.som.distance_map().T) # plotting the distance map as background
	#f.colorbar(a)
	t = np.zeros(len(self.target),dtype=int)
	t[self.target == 'A'] = 0
	t[self.target == 'B'] = 1
	t[self.target == 'C'] = 2
	t[self.target == 'D'] = 3

	# use different colors and markers for each label
	markers = ['o','s','D', '+']
	colors = ['r','g','b', 'y']
	for cnt,xx in enumerate(data):
	  w = self.som.winner(xx) # getting the winner
	  # place a marker on the winning position for the sample xx
	  tmp = self.axes.plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None',
	      markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)
	self.axes.axis([0,self.som.weights.shape[0],0,self.som.weights.shape[1]])
	#show() # show the figure
	#print "drawing"
	#self.figure.canvas.draw()



    def init_som(self, widget=None, data=None):
      ##print self.data
      ### Initialization and training ###
      cols = self.columns[self.combobox.get_active()]
      data = self.data[:, 0:len(cols)]

      #print len(cols)
      self.som = MiniSom(self.width_spin_button.get_value_as_int(), self.height_spin_button.get_value_as_int(), len(cols),sigma=1.2,learning_rate=0.5)
#      self.som.weights_init_gliozzi(data)
      self.som.random_weights_init(data)

    def train_som(self):
      cols = self.columns[self.combobox.get_active()]
      data = self.data[:, 0:len(cols)]
      print("Training...")
      #self.som.train_gliozzi(data) # Gliozzi et al training

      self.som.train_random(data,20)


      print("\n...ready!")

    def make_treeview(self, data, liststore):
	#i = 0
	cols = self.columns[self.combobox.get_active()]
	#print type(cols)
	#print len(cols)
	for d in data:
	  #i += 1

	  tmp = d.tolist()
	  #print 'tmp', tmp
	  #while len(tmp) < cols:
	    #tmp.append(False)
	    #print 'tmp', tmp
	    #cols = cols - 1
	  Qe = MiniSom.quantization_error_subset(self.som,d,len(cols))
	  #print tmp
	  tmp.append(Qe)
	  tmp.append(4 * Qe ** 0.5)
	  liststore.append(tmp)

	treeview = gtk.TreeView(model=liststore)
	#i = 0
	for d in range(len(self.test_data[0])):
	  #print i
	  #i += 1
	  renderer_text = gtk.CellRendererText()
	  column_text = gtk.TreeViewColumn(self.pattern_labels[d], renderer_text, text=d)
	  treeview.append_column(column_text)
	column_text = gtk.TreeViewColumn('Qe', renderer_text, text=d+1)
	treeview.append_column(column_text)
	column_text = gtk.TreeViewColumn('NLT', renderer_text, text=d+2)
	treeview.append_column(column_text)

	return treeview

    def update_treeview(self, data, liststore):
      	cols = len(self.columns[self.combobox.get_active()])

	for i, d in enumerate(data):

	  for j in range(len(d)):
	    #print j

	    liststore[i][j] = d[j]

	    if j >= cols:
	      liststore[i][j] = -999
	  Qe = MiniSom.quantization_error_subset(self.som,d,cols)

	  #print d, liststore[i]
	  liststore[i][-2]= Qe
	  liststore[i][-1]= 4 * Qe ** 0.5

    def select_columns(self, widget=None):
      #self.open_file(self.file_name)
      #self.init_som()
      self.update_treeview(self.test_data, self.test_liststore)
      self.update_treeview(self.data, self.patterns_liststore)


#----------------------------------------
# SAM added these functions here

    def pertSomWeights( self,  widget=None, data=None ):
        #if scale == None:
        scale = .5
        print( 'Adding noise to SOM weights')
        # print( self.som.weights )
        # print( self.som.weights.shape )
	pertAmount = scale*(np.random.random_sample( self.som.weights.shape)-.5)
        self.som.weights = self.som.weights + pertAmount
#	print self.som.weights
	self.Draw_figure()
	self.canvas.draw()
	self.canvas.draw_idle()
	#We need to draw *and* flush
	self.figure.canvas.draw()
	self.figure.canvas.flush_events()


    def pertInputs( self,  widget=None, data=None ):
        #if scale == None:
        p = .2
        print( 'Making %f prop of inputs 0.5' %p)
        #print( self.data.shape )
	
        # randomly get indices to switch, then replace
	noiseIndex = np.random.binomial(1,p, self.data.shape)  #ones at p proportion of samples
	self.data[noiseIndex ==1 ] = .5
	print( self.data )
	# update the treeview for the "Patterns" tab to see the result graphically 
	self.update_treeview(self.data, self.patterns_liststore)


#----------------------------------------
    def __init__(self):
      # create a new window
      self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
      # When the window is given the "delete_event" signal (this is given
      # by the window manager, usually by the "close" option, or on the
      # titlebar), we ask it to call the delete_event () function
      # as defined above. The data passed to the callback
      # function is NULL and is ignored in the callback function.
      self.window.connect("delete_event", self.delete_event)
      # Here we connect the "destroy" event to a signal handler.
      # This event occurs when we call gtk_widget_destroy() on the window,
      # or if we return FALSE in the "delete_event" callback.
      self.window.connect("destroy", self.destroy)

      #window.set_icon_from_file(get_resource_path("icon.png"))
      #window.connect("delete-event", Quit)
      #window.connect("destroy", Quit)
      self.window.set_title("SOM model")
      self.window.set_default_size(500, 500) #this si to ensure the window is always the smallest it can be
      #self.window.set_resizable(False)
      #window.set_border_width(10)

      # Args are: homogeneous, spacing, expand, fill, padding
      homogeneous = False
      spacing = 0
      expand = False
      fill = False
      padding = 10

      self.hbox = gtk.HBox(homogeneous, spacing)
      self.vbox = gtk.VBox(homogeneous, spacing)
      self.window.add(self.vbox)


      #self.adjustment = gtk.Adjustment(value=10000, lower=1, upper=100000000, step_incr=1000, page_incr=10000)
      #self.iterations_spin_button = gtk.SpinButton(self.adjustment, climb_rate=0, digits=0)
      self.label = gtk.Label("Dimensions:")

      self.adjustment = gtk.Adjustment(value=5, lower=1, upper=100, step_incr=2, page_incr=5)
      self.width_spin_button = gtk.SpinButton(self.adjustment, climb_rate=0, digits=0)
      self.adjustment = gtk.Adjustment(value=10, lower=1, upper=100, step_incr=2, page_incr=5)
      self.height_spin_button = gtk.SpinButton(self.adjustment, climb_rate=0, digits=0)


      # Create a series of buttons with the appropriate settings

      image = gtk.Image()
      #  (from http://www.pygtk.org/docs/pygtk/gtk-stock-items.html)
      image.set_from_stock(gtk.STOCK_EXECUTE, 1)
      self.play = gtk.Button()
      self.play.set_image(image)
      self.play.set_label("Train")

      #image = gtk.Image()
      ##  (from http://www.pygtk.org/docs/pygtk/gtk-stock-items.html)
      #image.set_from_stock(gtk.STOCK_APPLY, 1)
      #self.test = gtk.Button()
      #self.test.set_image(image)
      #self.test.set_label("Test")

      image = gtk.Image()
      #  (from http://www.pygtk.org/docs/pygtk/gtk-stock-items.html)
      image.set_from_stock(gtk.STOCK_OPEN, 1)
      self.open = gtk.Button()
      self.open.set_image(image)
      self.open.set_label("Open patterns")

      #self.pause = gtk.Button(stock = gtk.STOCK_MEDIA_PAUSE)

      image = gtk.Image()
      image.set_from_stock(gtk.STOCK_REFRESH, 1)
      self.reset = gtk.Button()
      self.reset.set_image(image)
      self.reset.set_label("Reset")

      self.play.connect("clicked", self.Run, None)
      #self.test.connect("clicked", self.Test, None)
      self.open.connect("clicked", self.select_file, None)

      #self.pause.connect("clicked", self.Pause, None)
      self.reset.connect("clicked", self.Reset, None)
      self.height_spin_button.connect("value-changed", self.Reset, "Height changed")
      self.width_spin_button.connect("value-changed", self.Reset, "Width changed")

      # add perturb button to disturb trained som weights
      self.perturb = gtk.Button("Perturb SOM") # create gtk button to perturb som weights
      self.perturb.connect( "clicked", self.pertSomWeights, None ) # run self.pertSomWeights
      self.perturb.show() # tell GTK to show button, but not where
       
      # add button to add noisy encoding to training inputs
      self.perturbInputButton = gtk.Button("Perturb Inputs") # create gtk button to perturb som weights
      self.perturbInputButton.connect( "clicked", self.pertInputs, None ) # run self.pertSomWeights
      self.perturbInputButton.show() # tell GTK to show button, but not where
	


      #self.width_spin_button.connect("value_changed", self.init_som)
      #self.height_spin_button.connect("value_changed", self.init_som)

      #self.som = Environment(width = self.width_spin_button.get_value_as_int(), height = self.height_spin_button.get_value_as_int())
      #self.som.show()
      #self.pause.set_sensitive(self.som.paused)
      #self.vbox.pack_start(self.som, True, True, 0)
      #file_names =  #  ['stimuli.csv']

      allFileName = '4750.csv' #'stimuli.csv'	
      self.file_name =  allFileName  #'4749.csv' # 'stimuli.csv' # file_names[0]
      self.test_file_name = allFileName #'4749.csv' # 'stimuli.csv'

      self.visual_only = [0,1,2,3,4,5,6,7]
      self.visual_and_acoustic = [0,1,2,3,4,5,6,7,8]
      self.columns = [self.visual_only, self.visual_and_acoustic]

      
      #f = Figure(figsize=(5,4), dpi=100)
      #a = f.add_subplot(111)
      self.combobox = gtk.combo_box_new_text()
      self.combobox.append_text('Visual only')
      self.combobox.append_text('Visual and acoustic')
      self.test_data = np.genfromtxt(self.test_file_name, delimiter=',',usecols=(self.visual_and_acoustic),skip_header=1)
      self.test_data +=  -.5 #0.00001



      self.test_data = np.apply_along_axis(lambda x: x/np.linalg.norm(x),1,self.test_data) # data normalization

      self.target = np.genfromtxt(self.file_name,delimiter=',',usecols=(9),dtype=str,skip_header=1) # loading the labels for use in the figure
      self.combobox.set_active(1)
      self.combobox.connect('changed', self.Reset)
      #cols = self.columns[self.combobox.get_active()]
      #print cols
      self.data = np.genfromtxt(self.file_name, delimiter=',',usecols=(self.visual_and_acoustic),skip_header=1)
      self.data += -.5  #0.00001
      self.data = np.apply_along_axis(lambda x: x/np.linalg.norm(x),1,self.data) # data normalization

      #self.pattern_labels = np.genfromtxt(self.file_name, delimiter=',',usecols=(self.visual_and_acoustic), skip_footer=14, dtype=str)
      self.pattern_labels = np.genfromtxt(self.file_name, delimiter=',',usecols=(self.visual_and_acoustic), dtype=str)[0]


      #print self.pattern_labels
      self.init_som()
      #self.toolbar = NavigationToolbar(self.canvas, self.window)
      #self.vbox.pack_start(self.toolbar, False, False)
      #self.vbox.pack_start(self.canvas)
      self.test_liststore = gtk.ListStore(float, float, float, float, float, float, float, float, float, float, float)
      self.patterns_liststore = gtk.ListStore(float, float, float, float, float, float, float, float, float, float, float)

      self.test_treeview = self.make_treeview(self.test_data, self.test_liststore)
      self.patterns_treeview = self.make_treeview(self.data, self.patterns_liststore)
      #self.data = np.genfromtxt(self.file_name, delimiter=',',usecols=(0,1,2,3,4,5,6,7),skip_header=1)
      #self.pattern_labels = np.genfromtxt(self.file_name, delimiter=',',usecols=(0,1,2,3,4,5,6,7), skip_footer=8, dtype=str)
      ##self.data = np.apply_along_axis(lambda x: x/np.linalg.norm(x),1,self.data) # data normalization





      self.figure, self.axes= plt.subplots()

      # Create canvas.
      self.canvas = FigureCanvas(self.figure)  # a gtk.DrawingArea
      self.canvas.set_size_request(300, 400)
      self.Draw_figure()





      self.notebook = gtk.Notebook()
      self.notebook.set_tab_pos(gtk.POS_TOP)
      self.vbox.pack_start(self.notebook)

      label = gtk.Label("Distance map")
      self.notebook.append_page(self.canvas, label)
      label = gtk.Label("Patterns")
      self.notebook.append_page(self.patterns_treeview, label)
      label = gtk.Label("Testing")
      #hbox = gtk.HBox(homogeneous, spacing)

      self.notebook.append_page(self.test_treeview, label)
      #hbox.pack_start(test_treeview, expand, fill, 0)
      #hbox.pack_start(test_treeview, expand, fill, 0)


      self.patterns_treeview.show()
      self.test_treeview.show()


      self.canvas.draw_idle()
      self.canvas.show()
      self.figure.canvas.draw()

      self.vbox.pack_start(self.hbox, expand, fill, 10)
      self.status_bar = gtk.Statusbar()
      self.vbox.pack_start(self.status_bar, expand, fill, 0)
      self.status_bar.show()
      glib.idle_add(self.Status_update)
      self.hbox.show()
      self.vbox.show()
      self.play.show()
      #self.test.show()
      self.open.show()

      #self.pause.show()
      self.reset.show()
      #self.iterations_spin_button.show()
      self.width_spin_button.show()
      self.height_spin_button.show()



      self.hbox.pack_start(self.play, expand, fill, padding)
      #self.hbox.pack_start(self.test, expand, fill, padding)
      self.hbox.pack_start(self.open, expand, fill, padding)
      self.hbox.pack_start(self.combobox, expand, fill, padding)
      #self.hbox.pack_start(self.pause, expand, fill, 0)
      self.hbox.pack_start(self.reset, expand, fill, padding)
      #self.hbox.pack_start(self.iterations_spin_button, expand, fill, 0)
      self.hbox.pack_start(self.label, expand, fill, padding)

      self.hbox.pack_start(self.width_spin_button, expand, fill, padding)
      self.hbox.pack_start(self.height_spin_button, expand, fill, 0)
      self.hbox.pack_start( self.perturb, expand, fill, padding)
      self.hbox.pack_start( self.perturbInputButton, expand, fill, padding)

	


      #self.quit = gtk.Button("Quit")
      self.quit = gtk.Button(stock = gtk.STOCK_QUIT)
      self.combobox.connect('changed', self.select_columns)

      self.quit.connect("clicked", self.destroy, None)
      self.hbox.pack_end(self.quit, expand, fill, padding)
      self.quit.show()
      #print window.get_size()





      self.window.show_all()



      self.window.present()
      #gtk.main()
      # And of course, our main loop.
      #gtk.main()
      # Control returns here when main_quit() is called


      return None

    def main(self):

    # All PyGTK applications must have a gtk.main(). Control ends here
    # and waits for an event to occur (like a key press or mouse event).
      gtk.main()






##----------------------------------------------------------------------------------------------------##
# If the program is run directly or passed as an argument to the python
# interpreter then create a HelloWorld instance and show it
if __name__ == "__main__":
    model = SOM()
    model.main()

