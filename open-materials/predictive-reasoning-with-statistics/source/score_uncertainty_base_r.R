# Independent base-R path: no package, refitting, RNG, or Python helper.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2)
resource <- args[1]; out <- args[2]
read <- function(name) read.csv(file.path(resource,name),stringsAsFactors=FALSE,check.names=FALSE)
data <- read('data/unit07_rows.csv')
data <- data[data$role=='temporal_final',]
data <- data[order(data$row_id),]
stopifnot(nrow(data)==1341,length(unique(data$row_id))==1341,all(data$y %in% c(0,1)))
colleges <- sort(unique(data$college_id));stopifnot(length(colleges)==16)
pred <- read('predictions/unit07_predictions.csv')
probability <- function(model) {
  d <- pred[pred$model_id==model,]; d <- d[order(d$row_id),]
  stopifnot(identical(d$row_id,data$row_id),identical(d$y,data$y),all(is.finite(d$probability)),all(d$probability>=0 & d$probability<=1))
  d$probability
}
full <- probability('full_logistic')
assignment <- read('splits/score_college_bootstrap.csv')
draws <- as.matrix(assignment[,paste0('slot_',sprintf('%02d',1:16))])
stopifnot(all(draws %in% colleges))
size <- table(factor(data$college_id,levels=colleges))
college_rows <- list(); draw_rows <- list(); summaries <- list()
for (model in c('control_mean','control_table')) {
  loss <- (data$y-probability(model))^2-(data$y-full)^2
  totals <- tapply(loss,factor(data$college_id,levels=colleges),sum)
  d <- apply(draws,1,function(selected) sum(totals[selected])/sum(size[selected]))
  n <- apply(draws,1,function(selected) sum(size[selected]))
  bounds <- quantile(d,c(.025,.975),type=7,names=FALSE)
  college_rows[[model]] <- data.frame(comparator=model,college_id=colleges,n=as.integer(size),difference_sum=as.numeric(totals))
  draw_rows[[model]] <- data.frame(comparator=model,replicate=assignment$replicate,n=n,difference=d)
  summaries[[model]] <- data.frame(comparator=model,n=1341,colleges=16,replicates=nrow(assignment),difference=mean(loss),lower=bounds[1],upper=bounds[2])
}
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write.csv(do.call(rbind,college_rows),file.path(out,'score_colleges.csv'),row.names=FALSE)
write.csv(do.call(rbind,draw_rows),file.path(out,'score_draws.csv'),row.names=FALSE)
write.csv(do.call(rbind,summaries),file.path(out,'score_summary.csv'),row.names=FALSE)
