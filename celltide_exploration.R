library(tidyverse)
library(directlabels)

x <- read_csv("test_tide.csv")
markers <- read_csv("exemplar-001/markers.csv")

y <- x %>%
  mutate(
    cell_id = row_id + 1
  ) %>%
  select(-row_id) %>%
  pivot_longer(
    c(-tide_radius, -cell_id),
    names_to = c("measure", "channel"),
    names_sep = "-",
    values_to = "value"
  ) %>%
  mutate(
    channel = as.integer(channel) + 1L,
    value_norm = if_else(
      measure %in% c("area"),
      value,
      log2(value)
    )
  ) %>%
  left_join(
    select(markers, channel_number, marker_name),
    by = c("channel" = "channel_number")
  )

p <- y %>%
  filter(
    cell_id %in% c(1, 2, 3),
    !replace_na(str_detect(marker_name, "AF[0-9]{3}"), FALSE),
    !replace_na(str_detect(marker_name, "background"), FALSE)
  ) %>%
  ggplot(aes(tide_radius, value_norm, color = marker_name)) +
    geom_line() +
    facet_grid(vars(measure), vars(cell_id), scales = "free_y") +
    coord_cartesian(xlim = c(-8, 4))

direct.label(p, method = "first.bumpup")
