# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 16:17:38 2021

@author: trang.le
VTK tutorials
"""
import vtk


def main(argv):
    #
    # Next we create an instance of vtkNamedColors and we will use
    # this to select colors for the object and background.
    #
    colors = vtk.vtkNamedColors()

    #
    # Now we create an instance of vtkConeSource and set some of its
    # properties. The instance of vtkConeSource "cone" is part of a
    # visualization pipeline (it is a source process object) it produces data
    # (output type is vtkPolyData) which other filters may process.
    #
    cone = vtk.vtkConeSource()
    cone.SetHeight(3.0)
    cone.SetRadius(1.0)
    cone.SetResolution(10)

    #
    # In this example we terminate the pipeline with a mapper process object.
    # (Intermediate filters such as vtkShrinkPolyData could be inserted in
    # between the source and the mapper.)  We create an instance of
    # vtkPolyDataMapper to map the polygonal data into graphics primitives. We
    # connect the output of the cone source to the input of this mapper.
    #
    coneMapper = vtk.vtkPolyDataMapper()
    coneMapper.SetInputConnection(cone.GetOutputPort())

    #
    # Create an actor to represent the cone. The actor orchestrates rendering
    # of the mapper's graphics primitives. An actor also refers to properties
    # via a vtkProperty instance, and includes an internal transformation
    # matrix. We set this actor's mapper to be coneMapper which we created
    # above.
    #
    coneActor = vtk.vtkActor()
    coneActor.SetMapper(coneMapper)
    coneActor.GetProperty().SetColor(colors.GetColor3d("MistyRose"))

    #
    # Create the Renderer and assign actors to it. A renderer is like a
    # viewport. It is part or all of a window on the screen and it is
    # responsible for drawing the actors it has.  We also set the background
    # color here.
    #
    ren1 = vtk.vtkRenderer()
    ren1.AddActor(coneActor)
    ren1.SetBackground(colors.GetColor3d("MidnightBlue"))

    ren1.SetViewport(0.0, 0.0, 0.5, 1.0)

    ren2 = vtk.vtkRenderer()
    ren2.AddActor(coneActor)
    ren2.SetBackground(colors.GetColor3d("DodgerBlue"))
    ren2.SetViewport(0.5, 0.0, 1.0, 1.0)
    # Finally we create the render window which will show up on the screen.
    # We put our renderer into the render window using AddRenderer. We also
    # set the size to be 300 pixels by 300.
    #
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren1)
    renWin.AddRenderer(ren2)
    renWin.SetSize(600, 300)
    renWin.SetWindowName("Tutorial_Step1")

    #
    # Make one view 90 degrees from other.
    #
    ren1.ResetCamera()
    ren1.GetActiveCamera().Azimuth(90)

    #
    # Now we loop over 360 degrees and render the cone each time.
    #
    for i in range(0, 360):
        # Render the image
        renWin.Render()
        # Rotate the active camera by one degree.
        ren1.GetActiveCamera().Azimuth(1)
        ren2.GetActiveCamera().Azimuth(1)


class vtkMyCallback(object):
    """
    Callback for the interaction.
    """

    def __init__(self, renderer):
        self.renderer = renderer

    def __call__(self, caller, ev):
        position = self.renderer.GetActiveCamera().GetPosition()
        print("({:5.2f}, {:5.2f}, {:5.2f})".format(*position))


if __name__ == "__main__":
    import sys

    main(sys.argv)
