bgms_style <- function() {
  style <- styler::tidyverse_style()
  # style$space$spacing_before_comments()
  # "if (condition)" -> "if(condition)"
  style$space$add_space_after_for_if_while <- function(pd_flat) {
    paren_after <- pd_flat$token %in% c("FOR", "IF", "WHILE")
    if (!any(paren_after)) {
      return(pd_flat)
    }
    pd_flat$spaces[paren_after & (pd_flat$newlines == 0L)] <- 0L
    pd_flat
  }

  # "a <- 1" -> "a = 1"
  style$token$force_assignment_op <- function(pd) {
    to_replace <- pd$token == "EQ_ASSIGN"
    pd$token[to_replace] <- "LEFT_ASSIGN"
    pd$text[to_replace] <- "="
    pd
  }
  style
}

# run the styler locally via
# note that this cannot be undone! make sure to commit your changes beforehand
# source("inst/styler/bgms_style.R")
# styler::style_pkg(
#   style = bgms_style,
#   exclude_files = c("R/RcppExports\\.R", "R/cpp11\\.R", "R/import-standalone.*\\.R", "\\.Rprofile\\.R")
# )
