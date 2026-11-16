# Student guide

The same project supports [Blindjack](../labs/blindjack/README.md),
[Line Following](../labs/line_following/README.md), and
[Visual Queries](../labs/visual_queries/README.md). The numbered commands, working
directories, submission format and AI-use policy are shared. Observations, actions,
training methods and scores are defined by each lab's statement.

| Task | Read or open |
| --- | --- |
| Install, choose a lab and run a template | [Project quick start](../README.md) |
| Write, train and debug a solution | [Development](development.md) |
| Inspect data or controller behaviour | [Graphical tools](development.md#graphical-inspection-and-manual-play) |
| Understand image questions, adaptation and scoring | [Visual Queries statement](../labs/visual_queries/README.md) |
| Read question tables, PNG sheets and vocabulary files | [Visual Queries data format](../labs/visual_queries/DATA.md) |
| Explore visual and language modelling approaches | [Visual Queries hints](../labs/visual_queries/HINTS.md) |
| Implement numeric learning and prediction diagnostics | [Visual Queries API](api-v3.md) |
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
hand-tuned controller; its preparation procedure can be replaced with learning or
fitting. Visual Queries' baseline learns from image/question pairs and adapts to
new vocabulary. Each statement identifies the supplied settings and measured
examples; the instructor announces the assessment profile.
