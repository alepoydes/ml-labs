# Student guide

The same project supports [Blindjack](../labs/blindjack/README.md) and
[Line Following](../labs/line_following/README.md). The numbered commands, working
directories, submission format and AI-use policy are shared. Observations, actions,
training methods and scores are defined by each lab's statement.

| Task | Read or open |
| --- | --- |
| Install, choose a lab and run a template | [Project quick start](../README.md) |
| Write, train and debug a solution | [Development](development.md) |
| Inspect data or controller behaviour | [Graphical tools](development.md#graphical-inspection-and-manual-play) |
| Understand the robot task, data and API | [Line Following statement](../labs/line_following/README.md) |
| Study robot mechanics and numerical integration | [Robot dynamics](../labs/line_following/dynamics.md) |
| Learn about balancing, line following and reinforcement learning | [Control and learning](../labs/line_following/control_hints.md) |
| Browse robot courses and endpoint markers | [Graphical simulator and atlas](../labs/line_following/README.md#visualisation-and-debugging) |
| Inspect the Blindjack MLP's learning curves | [Blindjack notebook](notebook.md) |
| Understand local execution and grading limits | [Environment](environment.md) |
| Package and submit one variant | [Submission format](submissions.md) |
| Declare assistance | [AI use](ai-use.md) |
| See the remaining course topics | [Planned labs](planned-labs.md) |

Blindjack's baseline learns from labeled images. Line Following's baseline is a
hand-tuned controller: replace its preparation procedure with learning or fitting
for your own solution. Both labs currently use pilot settings.
