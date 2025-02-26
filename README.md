# boxey

**boxey** is a tool to efficiently create and run box models.

The concept of a box model is abstracted into three elements:
 - Compartments: are the "boxes", which track the conserved quantity's movement over time.
 - Processes: have an associated timescale and drive the movement of the conserved quantity both between Compartments and out of the system.
 - Inputs: define the flow of the conserved quantity into the system from external sources, and they potentially change over time.
**boxey** provides an easy API for building and running a model using these elements, and a consise, human-readable and boxey-readable format to make sharing models easy/encouraged.

## Examples

[Notebook](example.ipynb)