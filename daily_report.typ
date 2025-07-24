#set page(margin: (x: 0.8in, y: 0.8in))
#set text(font: "Times New Roman", size: 11pt)
#set par(justify: false)

// Header table with border
#table(
  columns: (auto, 1fr),
  stroke: 1pt,
  align: (center, center),

  // Logo cell
  table.cell(rowspan: 1, [
    #box(width: 1.2in, height: 1.2in)[
      //#image("anits_logo.png", width: 1.2in, height: 1.2in)
    ]
  ]),

  // Institution details cell
  [
    #text(size: 14pt, weight: "bold")[
      ANIL NEERUKONDA INSTITUTE OF TECHNOLOGY AND SCIENCES
    ]

    #v(0.1em)
    #text(size: 11pt, style: "italic")[
      (Affiliated to Andhra University)
    ]

    #v(0.1em)
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

#v(0.5em)

// Project details
#text(weight: "bold")[Project Batch No:]

#v(0.2em)

#text(weight: "bold")[Project Guide:]
Dr. D. Naga Teja
#v(0.2em)

#text(weight: "bold")[List of Members:]
#v(0.05em)
D. Sai Venkata Chaitanya (A22126510144)
#v(0.05em)
B. Harsith Veera Charan (A22126510134)
#v(0.05em)
Wuna Akhilesh (A22126510194)
#v(0.05em)
M. Sai Teja (A22126510163)
#v(0.05em)
Venkata Vishaal Tirupalli (A22126510193)

#v(0.2em)

#text(weight: "bold")[Project Title:]

#v(0.2em)

#text(size: 12pt, weight: "bold")[Diary of B.Tech Project work]

#v(0.1em)

#text(size: 11pt)[(in Project consultation hours , Project LAB and Other free hours)]

#v(0.5em)

// Main table
#table(
  columns: (auto, auto, auto, auto, auto, auto),
  stroke: 1pt,
  align: center + horizon,

  // Header row
  table.header(
    [#text(weight: "bold")[#h(1.5em) Date #h(1.5em)]],
    [#text(weight: "bold")[Time in Time Out]],
    [#text(weight: "bold")[Project Batch List]],
    [#text(weight: "bold")[Reg no of Students who are present]],
    [#text(weight: "bold")[Discussion Contents]],
    [#text(weight: "bold")[Signature of Supervisor]],
  ),

  // Empty data rows
  [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)],
  [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)],
  [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)],
  [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)],
  [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)],
  [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)],
  [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)], [#v(2em)],
)

#v(1em)

// Footer
#grid(
  columns: (1fr, 1fr),
  align: (left, right),
  [#text(weight: "bold")[Project Coordinator]], [#text(weight: "bold")[HOD-CSE]],
)
