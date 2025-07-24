#set page(margin: (x: 0.8in, y: 0.8in))
#set text(font: "Times New Roman", size: 10pt)
#set par(justify: false)

// Header table with institution details
#table(
  columns: (auto, 1fr),
  stroke: 1pt,
  align: (center, center),

  // Logo cell
  table.cell(rowspan: 1, [
    #box(width: 1.2in, height: 1.2in)[
      #image("anits_logo.png", width: 1.2in, height: 1.2in)
    ]
  ]),

  // Institution details cell
  [
    #text(size: 14pt, weight: "bold")[
      ANIL NEERUKONDA INSTITUTE OF TECHNOLOGY AND SCIENCES
    ]

    #v(0.3em)
    #text(size: 11pt, style: "italic")[
      (Affiliated to Andhra University)
    ]

    #v(0.5em)
    #text(size: 11pt)[
      Sangivalasa-531 162, BheemunipatnamMandal,

      Visakhapatnam Dt.
    ]
  ],
)

#v(1em)

// Department heading
#text(size: 12pt, weight: "bold")[
  DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
]

#v(1em)

// Title
#text(size: 12pt, weight: "bold")[
  B.Tech Project Proposal / Idea Submission Form (Review-1)
]

#v(1em)

// Header info
#grid(
  columns: (1fr, 1fr),
  [#text(weight: "bold")[Department of:] #h(2em) #line(length: 3in, stroke: 0.5pt)],
  [#text(weight: "bold")[Date:] #h(2em) #line(length: 2in, stroke: 0.5pt)],
)

#v(0.5em)

#text(weight: "bold")[Academic Year:] #h(2em) #line(length: 2in, stroke: 0.5pt)

#v(1em)

// Main form table
#table(
  columns: (2fr, 1fr, 1fr),
  stroke: 1pt,
  align: left + top,

  // Title row
  table.cell(colspan: 3, [#text(weight: "bold")[Title of the Project:] #v(1em)]),

  // Project type row
  table.cell(
    colspan: 3,
    [
      #text(weight: "bold")[
        What type of project it is? (#text(style: "italic")[Interdisciplinary / Society Application / Non society application / other]. Please specify)
      ]
      #v(2em)
    ],
  ),

  // Team lead info
  [#text(weight: "bold")[Team Lead Name:] #v(0.05em)
    B. Harsith Veera Charan (A22126510134)
    #v(0.05em)
  ],
  [#text(weight: "bold")[Team Lead Email:] #v(1.5em)],
  [#text(weight: "bold")[Team Lead Phone:] #v(1.5em)],

  // Team members info
  [#text(weight: "bold")[Team Members Name:]  #v(0.05em)
    D. Sai Venkata Chaitanya (A22126510144)
    #v(0.05em)
    Wuna Akhilesh (A22126510194)
    #v(0.05em)
    M. Sai Teja (A22126510163)
    #v(0.05em)
    Venkata vishaal Tirupalli (A22126510193)],
  [#text(weight: "bold")[Team Members Email:] #v(1.5em)],
  [#text(weight: "bold")[Team Members Phone:] #v(1.5em)],

  // Project proposal type
  [#text(weight: "bold")[Project proposal as part of:] Academic Requirement / Study Project #v(1.5em)],
  table.cell(colspan: 2, [#text(weight: "bold")[Innovation Type:] #v(1.5em)]),

  // Question 1
  table.cell(colspan: 3, [#text(weight: "bold")[1. Theme :] #v(3em)]),

  // Question 2
  table.cell(
    colspan: 3,
    [
      #text(weight: "bold")[
        2. Define the problem and its relevance to today's market / society / industry need (Describe how your idea could reach a significant number of end-users?):
      ]
      #v(4em)
    ],
  ),

  // Question 3
  table.cell(colspan: 3, [
    #text(weight: "bold")[
      3. Provide relevant background information and cite existing evidence that links to or supports your idea.
    ]
    #v(4em)
  ]),

  // Question 4
  table.cell(colspan: 3, [
    #text(weight: "bold")[
      4. Describe the Solution / Proposal :
    ]
    #v(4em)
  ]),

  // Question 5
  table.cell(colspan: 3, [
    #text(weight: "bold")[
      5. Describe your idea, including the methods and technologies involved.
    ]
    #v(4em)
  ]),

  // Question 6
  table.cell(
    colspan: 3,
    [
      #text(weight: "bold")[
        6. Explain the uniqueness and distinctive features of the (product / process / service) solution (Describe how your solution is innovative) :
      ]
      #v(4em)
    ],
  ),

  // Question 7
  table.cell(
    colspan: 3,
    [
      #text(weight: "bold")[
        7. How your proposed / developed (product / process / service) solution is different from similar kind of product by the competitors if any:
      ]
      #v(4em)
    ],
  ),

  // Question 8
  table.cell(colspan: 3, [
    #text(weight: "bold")[
      8. Is it Patentable project/Solution?: Yes/No
    ]
    #v(2em)
  ]),

  // Question 9
  table.cell(colspan: 3, [
    #text(weight: "bold")[
      9. Is the Solution commercializable either through Technology Transfer or Enterprise Development/Startup?: Yes/No
    ]
    #v(2em)
  ]),
)

#v(1em)

// Note
#text(style: "italic", weight: "bold")[
  **Attach additional sheets to this form to answer above questions , if necessary
]

#v(2em)

// Department section
#text(style: "italic", weight: "bold")[
  #underline[To be filled by Department Project Committee (DPC) only]
]

#v(1em)

#text(weight: "bold")[
  #underline[Approval Status:] #text(style: "italic")[Approved / Sent back for Modification]
]

#v(1em)

#text(weight: "bold")[
  #underline[Remarks / suggestions:]
]

#v(2em)
#line(length: 100%, stroke: 0.5pt)
#v(1em)
#line(length: 100%, stroke: 0.5pt)

#v(2em)

// Signature section
#grid(
  columns: (1fr, 1fr, 1fr),
  align: (center, center, center),
  [#text(weight: "bold")[Signature of\ Project Coordinator]],
  [#text(weight: "bold")[Signature of\ DPC Member]],
  [#text(weight: "bold")[Signature of\ HOD]],
)