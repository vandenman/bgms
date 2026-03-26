---
name: Bug report
about: Report something that is not working as expected.
title: ''
labels: bug
assignees: ''
---

## What happened?

Describe what went wrong. Be as specific as you can.

## Which function were you using?

For example: `bgm()`, `bgmCompare()`, `simulate_mrf()`, `predict()`, or one
of the extractor functions.

## Minimal example

If you can, paste a short R script that reproduces the problem. Starting from
one of the built-in datasets (`Wenchuan`, `Boredom`, `ADHD`) or simulated data
makes it easier for us to investigate.

```r
library(bgms)

# your code here
```

If the problem only occurs with your own data and you cannot share it, that is
fine — describe the data instead: number of variables, number of observations,
variable types (binary, ordinal, continuous), and any special characteristics
such as missing values or low-frequency categories.

If the code produces an error message, paste the full message below.

```
# error output here
```

## What did you expect to happen?

Describe the result you expected instead.

## Session information

Paste the output of `sessionInfo()` below. This tells us your R version,
operating system, and which packages are loaded — all of which help us
reproduce the problem.

To get this, run the following in your R console and copy the output:

```r
sessionInfo()
```

```
# paste sessionInfo() output here
```

## Anything else?

Add any other context that might help: screenshots, related issues, or
links to documentation.
