# data.py

# Define course data as a list of courses, where each course has topics
course_data = [
    {
        "name": "Human Computer Interaction",
        "code": "SCS 3209",
        "topics": [
            {
                "name": "Introduction to Human-Computer Interactions",
                "subtopics": [
                    "Importance of Human‐Computer Interaction",
                    "Components of HCI Model",
                    "What is an Interface?",
                    "Risk of Poor User Interface",
                    "Developing Interaction",
                    "HCI as a discipline and its short history",
                ],
                "duration": 1,  # Duration in hours
            },
            {
                "name": "Evolving Technologies for Rich Interaction",
                "subtopics": [
                    "Interactivity",
                    "Richer interaction",
                    "Multimodal and natural interaction",
                    "Gesture-based interaction",
                    "Effect of computing power for HCI",
                ],
                "duration": 1,  # Duration in hours
            },
            {
                "name": "HCI Paradigms and Metaphors",
                "subtopics": [
                    "Batch processing",
                    "Timesharing",
                    "Networking",
                    "Graphical display",
                    "Microprocessor",
                    "WWW",
                    "Ubiquitous computing",
                ],
                "duration": 2,  # Duration in hours
            },
            {
                "name": "Frameworks and Models in HCI",
                "subtopics": [
                    "Interaction Models",
                    "Human Error",
                    "Two gulfs in the interaction",
                    "Ergonomics",
                    "Interaction styles",
                ],
                "duration": 2,  # Duration in hours
            },
            {
                "name": "Interaction Design",
                "subtopics": [
                    "Golden rules of design",
                    "Navigation design",
                    "Screen design and layout",
                    "User action and control",
                ],
                "duration": 2,  # Duration in hours
            },
            {
                "name": "PACT Analysis",
                "subtopics": [
                    "PACT Framework for design feasibility",
                    "PACT Components",
                    "Task Analysis",
                ],
                "duration": 4,  # Duration in hours
            },
            {
                "name": "User Centered Design",
                "subtopics": [
                    "History",
                    "User-Centered Design (UCD)",
                    "Process of UCD",
                    "Mental Model and User Behavior",
                    "Persona and Scenario",
                    "Co-design",
                    "Participatory Design",
                ],
                "duration": 6,  # Duration in hours
            },
            {
                "name": "Accessibility, Usability and Universal Design",
                "subtopics": [
                    "Defining usability and its importance",
                    "5Es in Usability and Benefits",
                    "Human Interaction and Usability",
                    "Usability Heuristics",
                    "Accessibility and standards",
                    "Acceptability",
                    "General guidelines and principles",
                    "Universal Design",
                ],
                "duration": 4,  # Duration in hours
            },
            {
                "name": "Prototyping",
                "subtopics": [
                    "Overview of prototyping",
                    "Types of prototyping",
                    "Tools for prototyping",
                    "Developing a working prototype",
                ],
                "duration": 4,  # Duration in hours
            },
            {
                "name": "Implementation and Evaluation",
                "subtopics": [
                    "Implementation",
                    "Evaluation",
                    "Goals of Evaluation",
                    "Evaluation Techniques",
                    "Choosing an Evaluation Method",
                    "Design-after-design",
                ],
                "duration": 2,  # Duration in hours
            },
            {
                "name": "Usability Heuristics",
                "subtopics": [
                    "Overview of prototyping",
                    "Types of prototyping",
                    "Tools for prototyping",
                    "Developing a working prototype",
                ],
                "duration": 4,  # Duration in hours
            },
            {
                "name": "TASK Analysis",
                "subtopics": [
                    "Overview of prototyping",
                    "Types of prototyping",
                    "Tools for prototyping",
                    "Developing a working prototype",
                ],
                "duration": 4,  # Duration in hours
            },
        ],
    },
    {
        "name": "Database",
        "code": "SCS 4201",
        "topics": [
            {
                "name": "Database Security",
                "subtopics": [
                    "Importance of Human‐Computer Interaction",
                    "Components of HCI Model",
                    "What is an Interface?",
                    "Risk of Poor User Interface",
                    "Developing Interaction",
                    "HCI as a discipline and its short history",
                ],
                "duration": 1,  # Duration in hours
            },
            {
                "name": "Indexing",
                "subtopics": [
                    "Interactivity",
                    "Richer interaction",
                    "Multimodal and natural interaction",
                    "Gesture-based interaction",
                    "Effect of computing power for HCI",
                ],
                "duration": 1,  # Duration in hours
            },
            {
                "name": "Query Optimization",
                "subtopics": [
                    "Batch processing",
                    "Timesharing",
                    "Networking",
                    "Graphical display",
                    "Microprocessor",
                    "WWW",
                    "Ubiquitous computing",
                ],
                "duration": 2,  # Duration in hours
            },
            {
                "name": "Transaction Management",
                "subtopics": [
                    "Interaction Models",
                    "Human Error",
                    "Two gulfs in the interaction",
                    "Ergonomics",
                    "Interaction styles",
                ],
                "duration": 2,  # Duration in hours
            },
            {
                "name": "Distributed Databases",
                "subtopics": [
                    "Golden rules of design",
                    "Navigation design",
                    "Screen design and layout",
                    "User action and control",
                ],
                "duration": 2,  # Duration in hours
            }
        ],
    },
    {
        "name": "Middleware",
        "code": "SCS 3203",
        "topics": [
            {
                "name": "Introduction to Distributed Systems",
                "subtopics": [
                    "Importance of Human‐Computer Interaction",
                    "Components of HCI Model",
                    "What is an Interface?",
                    "Risk of Poor User Interface",
                    "Developing Interaction",
                    "HCI as a discipline and its short history",
                ],
                "duration": 1,  # Duration in hours
            },
            {
                "name": "Introduction to Middleware",
                "subtopics": [
                    "Interactivity",
                    "Richer interaction",
                    "Multimodal and natural interaction",
                    "Gesture-based interaction",
                    "Effect of computing power for HCI",
                ],
                "duration": 1,  # Duration in hours
            },
            {
                "name": "Middleware and Architectural Patterns",
                "subtopics": [
                    "Batch processing",
                    "Timesharing",
                    "Networking",
                    "Graphical display",
                    "Microprocessor",
                    "WWW",
                    "Ubiquitous computing",
                ],
                "duration": 2,  # Duration in hours
            },
            {
                "name": "Middleware and Distributed Objects",
                "subtopics": [
                    "Interaction Models",
                    "Human Error",
                    "Two gulfs in the interaction",
                    "Ergonomics",
                    "Interaction styles",
                ],
                "duration": 2,  # Duration in hours
            },
            {
                "name": "CORBA",
                "subtopics": [
                    "Golden rules of design",
                    "Navigation design",
                    "Screen design and layout",
                    "User action and control",
                ],
                "duration": 2,  # Duration in hours
            },
            {
                "name": "ESB and Integration Patterns",
                "subtopics": [
                    "PACT Framework for design feasibility",
                    "PACT Components",
                    "Task Analysis",
                ],
                "duration": 4,  # Duration in hours
            },
            {
                "name": "Transaction Management Concepts",
                "subtopics": [
                    "History",
                    "User-Centered Design (UCD)",
                    "Process of UCD",
                    "Mental Model and User Behavior",
                    "Persona and Scenario",
                    "Co-design",
                    "Participatory Design",
                ],
                "duration": 6,  # Duration in hours
            },
            {
                "name": "Availability",
                "subtopics": [
                    "Defining usability and its importance",
                    "5Es in Usability and Benefits",
                    "Human Interaction and Usability",
                    "Usability Heuristics",
                    "Accessibility and standards",
                    "Acceptability",
                    "General guidelines and principles",
                    "Universal Design",
                ],
                "duration": 4,  # Duration in hours
            },
        ],
    },
    # Define data for other courses here...
]
