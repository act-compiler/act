# ASPLOS 2026 ACT Tutorial: End-to-End Compiler Infrastructure for Emerging AI Accelerators

## Introduction

Recent years have seen a proliferation of specialized AI accelerators -- proposed in both academia (e.g., Gemmini, FEATHER) and industry (e.g., Google TPU, Intel AMX) -- that depart significantly from traditional CPU/GPU architectures.
However, research on compiler and systems support for these accelerators remains sparse, largely due to the lack of mature open-source ML compiler infrastructures capable of targeting them from popular ML frameworks like PyTorch, and JAX.
Building such support involves considerable manual effort, slowing innovation and creating a gap between hardware and software research communities.

This tutorial introduces the **ACT (Accelerator Compiler Toolkit)**, an ecosystem that automatically generates complete ML compiler backends and essential software tooling from high-level ISA specifications of AI accelerators.

The ACT ecosystem consists of:

- **TAIDL (Tensor Accelerator ISA Definition Language)**: A Python-based DSL for specifying AI accelerator ISAs. TAIDL leverages tensor IRs like XLA-HLO to compactly and precisely model execution semantics of AI accelerator ISAs.
- **TAIDL-TO (Test Oracle) Generator**: Automatically generates scalable functional simulators just from TAIDL specifications, enabling correctness testing of the software stack. TAIDL-TOs are orders of magnitude faster than existing simulators.
- **ACT Backend Generator**: Automatically generates sound and complete ML compiler backends just from TAIDL specification. ACT backends match or outperform state-of-the-art expert-written libraries, while maintaining low compile times (<1 sec).
- **XLA Integration**: Enables end-to-end compilation from popular ML frameworks like JAX, TensorFlow, and PyTorch.

In this hands-on tutorial, attendees will learn to use the ACT ecosystem to program a custom AI accelerator from scratch.
Through **four hands-on exercises** and **two demonstrations**, you will:

1. **Demonstration 1**: See a quick walkthrough of the ACT ecosystem and its components
2. **Hands-on Exercise 1**: Specify a new accelerator ISA using TAIDL
3. **Hands-on Exercise 2**: Write custom accelerator kernels using the generated kernel programming API
4. **Hands-on Exercise 3**: Generate and build a complete ML compiler backend just from the ISA specification
5. **Demonstration 2**: See end-to-end integration with JAX through XLA compiler
6. **Hands-on Exercise 4**: Modify the ISA specification and see how the changes propagate through the entire software stack

The tutorial is designed for researchers, practitioners, and students interested in compiler design, programming languages, and AI/ML hardware.
By the end, participants will have hands-on experience with the complete ACT workflow and understand how to rapidly prototype ML compiler support for novel AI accelerator architectures.

## Agenda

Date: Mar 22, 2026 (Sunday)  
Time: 1:30 PM - 6:00 PM (UTC-4)  
Venue: Ohio Room, The Landing Hotel  
Location: 757 Casino Dr. Pittsburgh, PA, USA  
Prerequisites: Please bring your own laptop with a working installation of Docker and follow the [tutorial setup instructions](./setup.md).

**Contents and Timeline (tentative)**

| Time           | Topic                                                                         | Presenter            | Slide or Code                                       |
| -------------- | ----------------------------------------------------------------------------- | -------------------- | --------------------------------------------------- |
| 1:30 - 1:35 PM | Welcome and Introduction                                                      | Prof. Charith Mendis | [Guide: Setup Instructions](./setup.md)             |
| 1:35 - 1:45 PM | Tutorial Logistics                                                            | Devansh Jain         |
| 1:50 - 2:10 PM | Talk: Overview of ACT Ecosystem                                               | Prof. Charith Mendis | Slide will be uploaded after the tutorials         |
| 2:10 - 2:20 PM | Demonstration 1: Quick walkthrough of ACT Ecosystem                           | Devansh Jain         | [Demo: ACT-walkthrough](./demos/act-walkthrough.md) |
| 2:20 - 2:50 PM | Hands-on Exercise 1: Specifying a new Accelerator ISA                         | Devansh Jain         | [Guide: Hands-on (1)](./exercise1/README.md)        |
| 2:50 - 3:10 PM | Talk: Expressivity and Extensibility of TAIDL                                 | Marco Frigo          | Slide will be uploaded after the tutorials         |
| 3:10 - 3:30 PM | Hands-on Exercise 2: Writing custom Accelerator Kernels                       | Devansh Jain         | [Guide: Hands-on (2)](./exercise2/README.md)        |
| 3:30 - 4:00 PM | Coffee Break                                                                  |                      |
| 4:00 - 4:30 PM | Talk: Automatically Generating Compiler Backends just from ISA Specifications | Akash Pardeshi       | Slide will be uploaded after the tutorials         |
| 4:30 - 5:00 PM | Hands-on Exercise 3: Generating a Compiler Backend for a new Accelerator ISA  | Devansh Jain         | [Guide: Hands-on (3)](./exercise3/README.md)        |
| 5:00 - 5:10 PM | Demonstration 2: Integrating the new Accelerator Backend with XLA Compiler    | Devansh Jain         | [Demo: JAX-XLA-ACT](./demos/jax-integration.md)     |
| 5:10 - 5:40 PM | Hands-on Exercise 4: Tweaking the ISA and Open Discussion                     | Devansh Jain         | [Guide: Hands-on (4)](./exercise4/README.md)        |
| 5:40 - 6:00 PM | Q&A and Closing Remarks                                                       | Prof. Charith Mendis |

---

## Organizers

- **Prof. Charith Mendis** is an Assistant Professor in the Siebel School of Computing and Data Science at the University of Illinois Urbana-Champaign (UIUC). His broad research interests are at the intersection of compilers, program optimization and machine learning. He received his Ph.D. and Master’s from the Massachusetts Institute of Technology and his B.Sc. from the University of Moratuwa. He is the recipient of a DARPA Young Faculty Award, an NSF CAREER Award, the William A. Martin outstanding master’s thesis award at MIT and the university gold medal for his B.Sc. He has won numerous paper awards including a Distinguished Paper Award at POPL, a Best Student Paper Award at the IEEE BigData conference, an honorable mention for the Best Artifact Award at SIGMOD, a Best Paper Award at ML for Systems workshop at ISCA and an IEEE Top Picks Honorable Mention.
- **Devansh Jain** is a Ph.D. student in the Siebel School of Computing and Data Science at the University of Illinois Urbana-Champaign (UIUC), advised by Prof. Charith Mendis. His research interests lie in the field of programming languages, compilers & computer architecture, primarily domain-specific languages and architectures. His primary research objective is to develop a unified compiler infrastructure for architectures designed for accelerating tensor computations. He has authored/co-authored multiple papers at top-tier PL & systems venues, including POPL, OOPSLA, MICRO, ISPASS, with a Distinguished Paper Award at POPL’25.
- **Akash Pardeshi** is a M.S. student in the Department of Electrical and Computer Engineering at the University of Illinois Urbana-Champaign (UIUC), advised by Prof. Charith Mendis. His broad research interests are in compilers and computer architecture. His research focuses on techniques such as equality saturation and e-graph applications to ML compilers.
- **Marco Frigo** is a B.S. student in the Department of Electrical and Computer Engineering at the University of Illinois Urbana-Champaign (UIUC). His broad research interests are in compilers and computer architecture. His research focuses on developing software infrastructure for emerging AI accelerators.
