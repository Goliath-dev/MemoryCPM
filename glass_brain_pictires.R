matrix_file <- "E:\\Work\\MBS\\EEG\\Modelling\\Memory\\matrix_glass_brain.csv"
matrix <- t(as.matrix(read.csv(matrix_file)))
atlas_file <- "E:\\Work\\MBS\\EEG\\Modelling\\Memory\\Destrieux atlas.csv"
atlas = read.csv(atlas_file)
brainconn(atlas=atlas, conmat=matrix, view="top", labels = F, label.size = 5, edge.alpha = 0.5, show.legend = F, edge.width = 3)
ggsave("E:\\Work\\MBS\\EEG\\Modelling\\Memory\\Results\\ImgsForArticle\\picture.png", bg="white")