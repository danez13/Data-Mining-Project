# Project 1
# Check List
- [X] main file
  - [X] arguments
  - [ ] rule generation
  - [ ] generate output file
    - [ ] items.txt
    - [ ] rules.txt
    - [ ] info.txt
- [ ] Visualization file
- [ ] Report
- [ ] Peer Evaluation


## Objective
The purpose of the project is to write a program to generate all association rules whose support is greater than a user-supplied minimum support and whose confidence is greater than a user-supplied minimum confidence. You will implement the Apriori algorithm to extract valuable patterns from the data, such as frequently purchased items together, and generate association rules that can help businesses improve product placement, promotions, or recommendations. You should follow the processes we mentioned in our slides.

## Input file Format
The input file, `small.txt`, consists of a set of lines, each line containing two numbers. The first number is the transaction ID, and the second number is the item ID. The lines in the file are ordered in increasing transaction ID order. Note that a transaction will be derived by combining the item IDs of all the lines that correspond to the same transaction ID. The input file is provided.

## Team Responsibilities

I expect all of you to contribute and do your best for your team! All three (3) members of the team should help with the entirety of the project. Note that each team has its own dedicated Canvas page. Feel free to use this space to share your files, have discussions, or ask me and our grader any questions you might have. To help you manage the project’s work, each team will have two algorithm developers and one visualization specialist and one report writer.
* Algorithm Developer: Manages input and output files, implements the frequent itemset generation and rule generation.
* Visualization Specialist: Focuses on creating effective visualizations of the results, keeps track of team meetings and efforts, writes the project report, outlining the methodology, results, and collaboration of the team.

## Deliverables
1. Main python file for rule generation
    1. You need to submit a generate-rules<teamID>.py (e.g. generate-rules01.py) file that will implement one simple run of the rule generation process based on the following input arguments:
        * minsuppc (minimum support count),
        * minconf (minimum confidence; when minconf=-1, do not generate rules.),
        * input_file (input file name).
    2. After generating the rules for the given arguments, you need to generate three different output files.
        1. items<teamID>.txt: This output file will contain as many lines as the number of frequent itemsets, with their support count. The format of each line will be *exactly* as follows: `ITEMSETS|SUPPORT_COUNT|SUPPORT`
            * ITEMSETS correspond to the items present in the itemset in a space-delimited fashion. E.g., “item1 item2 item3|10|0.1”. Notice that there are no other spaces other than the ones that separate the items. 
        2. rules<teamID>.txt: This output file will contain as many lines as the number of high-confidence frequent rules that you found. The format of each line will be *exactly* as follows: `LHS|RHS|SUPPORT_COUNT|SUPPORT|CONFIDENCE|LIFT`
            * where both LHS (and RHS) will contain the items that make up the left- (and right-) hand side of the rule in a space-delimited fashion. E.g., “item1 item2|item3|200|0.3|0.1|0.1”. Notice that there are no other spaces other than the ones that separate the items. When minconf=-1, the file will not be generated.
        3. info<teamID>.txt: This file will have a line for each piece of information, including the input parameters. More specifically, it will need to include the following information:
            ```
            minsuppc:
            minconf:
            input file:
            Number of items:
            Number of transactions:
            The length of the longest transaction:
            Number of frequent 1-itemsets:
            Number of frequent 2-itemsets:
            ...
            Number of frequent 𝑘 –itemsets:
            Total number of frequent itemsets:
            Number of high-confidence rules:
            The rule with the highest confidence:
            The rule with the highest lift:
            Time in seconds to find the frequent itemsets:
            Time in seconds to find the confident rules:
            ``` 
2. Python file or notebook for visualization
    * You need to submit a generate-plots<teamID>.py (e.g. generate-plots01.py) file that will read the necessary output files from the previous step and generate the plots needed for the report. All the plots should have axis labels and titles. Note that: all the required plots/information need to be included in your report; having them only in the notebook is not enough.

3. Report and requested output files
    * Complete the report00.docx file provided. The file submitted should have the name report<teamID>.pdf, (e.g.report01.pdf). Some requested output files will also need to be submitted. Do not copy-paste them into the report.
4. Peer evaluations