# MICRO 2025 Tutorial: End-to-End Compiler Infrastructure for Emerging Tensor Accelerators

## Introduction

Recent years have seen a proliferation of specialized ML accelerators—proposed in both academia (e.g., Gemmini, FEATHER) and industry (e.g., Google TPU, Intel AMX)—that depart significantly from traditional CPU/GPU architectures.
However, research on compiler and systems support for these accelerators remains sparse, largely due to the lack of open-source compiler infrastructures capable of targeting them from modern ML frameworks like PyTorch, TensorFlow, and JAX.
Building such support typically involves considerable manual effort, slowing innovation and creating a gap between hardware and software research communities.

This tutorial introduces the **ACT (Accelerator Compiler Toolkit)**, an ecosystem that automatically generates complete compiler backends and essential software tooling from high-level ISA specifications of tensor accelerators.

The ACT ecosystem consists of:

- **TAIDL (Tensor Accelerator ISA Definition Language)**: A Python-based DSL for specifying tensor accelerator ISAs. TAIDL leverages tensor IRs like XLA-HLO to compactly and precisely model execution semantics of tensor accelerator ISAs.
- **TAIDL-TO (Test Oracle) Generator**: Automatically generates scalable functional simulators just from TAIDL specifications, enabling correctness testing of the software stack. TAIDL-TOs are orders of magnitude faster than existing simulators.
- **ACT Backend Generator**: Automatically generates sound and complete compiler backends just from TAIDL specification. ACT backends match or outperform state-of-the-art expert-written libraries, while maintaining low compile times (<1 sec).
- **XLA Integration**: Enables end-to-end compilation from popular ML frameworks like JAX, TensorFlow, and PyTorch.

In this hands-on tutorial, attendees will learn to use the ACT ecosystem to program a custom tensor accelerator from scratch.
Through **three hands-on exercises** and **two demonstrations**, you will:

1. **Demonstration 1**: See a quick walkthrough of the ACT ecosystem and its components
2. **Hands-on Exercise 1**: Specify a new accelerator ISA using TAIDL
3. **Hands-on Exercise 2**: Write custom accelerator kernels using the generated kernel programming API
4. **Hands-on Exercise 3**: Generate and build a complete compiler backend just from the ISA specification
5. **Demonstration 2**: See end-to-end integration with JAX through XLA compiler

The tutorial is designed for researchers, practitioners, and students interested in compiler design, programming languages, and AI/ML hardware.
By the end, participants will have hands-on experience with the complete ACT workflow and understand how to rapidly prototype compiler support for novel accelerator architectures.

## Agenda

Date: Oct 19, 2025 (Sunday)  
Time: 1:00 PM - 5:00 PM (UTC+9)  
Venue: Berkeley Suite, 36th Floor, Lotte Hotel Seoul  
Location: 30 Eulji-ro, Jung District, Seoul, South Korea  
Prerequisites: Please bring your own laptop with a working installation of Docker and follow the [tutorial setup instructions](./setup.md) before the tutorial.

**Contents and Timeline (tentative)**

| Time           | Topic                                                                         | Presenter                   | Slide or Code                                       |
| -------------- | ----------------------------------------------------------------------------- | --------------------------- | --------------------------------------------------- |
| 1:00 - 1:10 PM | Tutorial Logistics                                                            | Devansh Jain (UIUC)         | [Guide: Setup Instructions](./setup.md)             |
| 1:10 - 1:20 PM | Welcome and Introduction                                                      | Prof. Charith Mendis (UIUC) |                                                     |
| 1:20 - 1:40 PM | Talk: Overview of ACT Ecosystem                                               | Prof. Charith Mendis (UIUC) | Slides will be uploaded after the tutorials         |
| 1:40 - 1:50 PM | Demonstration 1: Quick walkthrough of ACT Ecosystem                           | Devansh Jain (UIUC)         | [Demo: ACT-walkthrough](./demos/act-walkthrough.md) |
| 1:50 - 2:20 PM | Hands-on Exercise 1: Specifying a new Accelerator ISA                         | Devansh Jain (UIUC)         | [Guide: Hands-on (1)](./exercise1/README.md)        |
| 2:20 - 2:40 PM | Talk: Expressivity and Extensibility of TAIDL                                 | Marco Frigo (UIUC)          | Slides will be uploaded after the tutorials         |
| 2:40 - 3:00 PM | Hands-on Exercise 2: Writing custom Accelerator Kernels                       | Devansh Jain (UIUC)         | [Guide: Hands-on (2)](./exercise2/README.md)        |
| 3:00 - 3:30 PM | Break                                                                         |                             |                                                     |
| 3:30 - 4:00 PM | Talk: Automatically Generating Compiler Backends just from ISA Specifications | Akash Pardeshi (UIUC)       | Slides will be uploaded after the tutorials         |
| 4:00 - 4:30 PM | Hands-on Exercise 3: Generating a Compiler Backend for a new Accelerator ISA  | Devansh Jain (UIUC)         | [Guide: Hands-on (3)](./exercise3/README.md)        |
| 4:30 - 4:40 PM | Demonstration 2: Integrating the new Accelerator Backend with XLA Compiler    | Devansh Jain (UIUC)         | [Demo: JAX-XLA-ACT](./demos/jax-integration.md)     |
| 4:40 - 5:00 PM | Q&A and Closing Remarks                                                       | Devansh Jain (UIUC)         |                                                     |

## Organizers

- **Prof. Charith Mendis** is an Assistant Professor in the Siebel School of Computing and Data Science at the University of Illinois at Urbana-Champaign. His broad research interests are at the intersection of compilers, program optimization and machine learning. He received his Ph.D. and Master’s from the Massachusetts Institute of Technology and his B.Sc. from the University of Moratuwa. He is the recipient of a DARPA Young Faculty Award, an NSF CAREER Award, the William A. Martin outstanding master’s thesis award at MIT and the university gold medal for his B.Sc. He has won numerous paper awards including a Distinguished Paper Award at POPL, a Best Student Paper Award at the IEEE BigData conference, an honorable mention for the Best Artifact Award at SIGMOD, a Best Paper Award at ML for Systems workshop at ISCA and an IEEE Top Picks Honorable Mention.
- **Devansh Jain** is a Ph.D. student in the Siebel School of Computing and Data Science at the University of Illinois Urbana-Champaign (UIUC), advised by Prof. Charith Mendis. His research interests lie in the field of programming languages, compilers & computer architecture, primarily domain-specific languages and architectures. His primary research objective is to develop a unified compiler infrastructure for architectures designed for accelerating tensor computations. He has authored/co-authored multiple papers at top-tier PL & systems venues, including POPL, OOPSLA, MICRO, with a Distinguished Paper Award at POPL’25.
- **Akash Pardeshi** is a M.S. student in the Department of Electrical and Computer Engineering at the University of Illinois Urbana-Champaign (UIUC), advised by Prof. Charith Mendis. His broad research interests are in compilers and computer architecture. His research focuses on techniques such as equality saturation and e-graph applications to ML compilers.
