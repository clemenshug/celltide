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


profile_res <- read_csv("profile_test_res.csv.gz")


profile_res_clean <- profile_res %>%
  group_by(cell_1, cell_2) %>%
  filter(if(any(is.na(value))) FALSE else TRUE) %>%
  ungroup() %>%
  mutate(
    position = position_id - 2
  ) %>%
  left_join(
    select(markers, marker_name) %>%
      mutate(marker_id = seq_len(n()) - 1),
    by = c("marker_id")
  )

profile_res_clean %>%
  filter(!str_starts(marker_name, "DNA")) %>%
  View()

profile_res_clean %>%
  filter(!str_starts(marker_name, "DNA")) %>%
  group_by(cell_1, cell_2, marker_name) %>%
  filter(all(value > 0)) %>%
  mutate(
    cv = sd(value) / mean(value)
  ) %>%
  ungroup() %>%
  arrange(desc(cv), cell_1, cell_2, marker_name, position) %>%
  View()

cell_profiles <- read_csv("test_out/cell_profiles.csv") %>%
  mutate(
    cell_id = row_id + 1
  ) %>%
  select(-row_id) %>%
  pivot_longer(
    c(-expansion_radius, -cell_id),
    names_to = c("measure", "channel"),
    names_sep = "-",
    values_to = "value"
  ) %>%
  mutate(
    channel = as.integer(channel) + 21L,
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

contact_profiles <- read_csv("test_out/contact_profiles.csv") %>%
  pivot_longer(
    c(-cell_1, -cell_2, -marker_id),
    names_to = "expansion_radius",
    values_to = "value"
  )  %>%
  mutate(
    across(expansion_radius, as.integer),
    channel = as.integer(marker_id) + 21L,
    value_norm = log2(value)
  ) %>%
  left_join(
    select(markers, channel_number, marker_name),
    by = c("channel" = "channel_number")
  )

library(broom)
absmax <- function(x) head(x[which.max(abs(x))], n = 1)

contact_profiles_summary <- contact_profiles %>%
  group_by(
    cell_1, cell_2, marker_id, marker_name, channel
  ) %>%
  summarize(
    max_diff = absmax(
      value_norm[expansion_radius > 0]
    ) - absmax(
      value_norm[expansion_radius < 0]
    ),
    response_model = possibly(lm, list(coefficients = list(expansion_radius=NA)))(
      value_norm ~ expansion_radius, data = cur_data()
    ) %>%
      # tidy() %>%
      list(),
    slope = response_model %>%
      map("coefficients") %>%
      map_dbl("expansion_radius"),
    .groups = "drop"
  )

cell_profiles_summary <- cell_profiles %>%
  filter(measure == "intensity_mean") %>%
  group_by(cell_id, measure, channel, marker_name) %>%
  summarize(
    max_diff = absmax(
      value_norm[expansion_radius > 0]
    ) - absmax(
      value_norm[expansion_radius < 0]
    ),
    response_model = possibly(lm, list(coefficients = list(expansion_radius=NA)))(
      value_norm ~ expansion_radius, data = cur_data()
    ) %>%
      # tidy() %>%
      list(),
    slope = response_model %>%
      map("coefficients") %>%
      map_dbl("expansion_radius"),
    intensity_mean = value_norm[expansion_radius == 0],
    .groups = "drop"
  )

coi <- c(7016, 6965, 7069)
coi <- coi <- c(7016, 6965, 7069) - 1
moi <- c("CD11B", "DNA_6")

cell_profiles %>%
  filter(
    cell_id %in% coi,
    marker_name %in% moi
  ) %>%
  ggplot(
    aes(expansion_radius, value_norm, color = as.factor(cell_id))
  ) +
    geom_line() +
    facet_wrap(~marker_name, scales = "free_y")

contact_profiles %>%
  filter(
    cell_1 %in% coi,
    cell_2 %in% coi,
    marker_name %in% moi
  ) %>%
  ggplot(
    aes(expansion_radius, value_norm, color = marker_name)
  ) +
  geom_line() +
  facet_wrap(~cell_1 + cell_2, scales = "free_y")



library(igraph)
library(tidygraph)
library(ggraph)

cell_profiles_summary_graph <- cell_profiles_summary %>%
  filter(
    cell_id %in% union(contact_profiles_summary$cell_1, contact_profiles_summary$cell_2)
  ) %>%
  select(
    cell_id, max_diff, slope, intensity_mean, marker_name
  ) %>%
  pivot_wider(names_from = marker_name, values_from = c(max_diff, slope, intensity_mean))

contact_profiles_summary_graph <- contact_profiles_summary %>%
  select(
    cell_1, cell_2, marker_name, max_diff, slope
  ) %>%
  pivot_wider(names_from = marker_name, values_from = c(max_diff, slope))

profile_graph <- graph_from_data_frame(
  contact_profiles_summary_graph,
  vertices = cell_profiles_summary_graph,
  directed = FALSE
)

ggraph(
  profile_graph, layout = "igraph", algorithm = "nicely"
) +
  geom_edge_fan(aes(color = slope_FOXP3)) +
  geom_node_point(aes(size = intensity_mean_FOXP3))
