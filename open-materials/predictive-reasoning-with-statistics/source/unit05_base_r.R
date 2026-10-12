#!/usr/bin/env Rscript
# Independent base-R joins, encoders, QR calibration and rank-pair scoring.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L) stop('usage: unit05_base_r.R RESOURCE OUTDIR')
root <- normalizePath(args[1],mustWork=TRUE); dest <- args[2]
options(digits=17); invisible(Sys.setlocale('LC_COLLATE','C'))
need <- function(ok,message) if(!isTRUE(ok)) stop(message)
read <- function(path) read.csv(path,colClasses='character',na.strings=NULL,check.names=FALSE)
raw <- file.path(root,'source/tonmick-v1.0.0/data/anchor')
p <- read(file.path(raw,'participants.csv')); b <- read(file.path(raw,'baseline.csv')); y <- read(file.path(raw,'registry.csv'))
for(tab in list(p,b,y)) need(!anyDuplicated(tab$student_id),'duplicate ID')
need(setequal(b$student_id,y$student_id) && setequal(b$student_id,p$student_id[p$responded_baseline=='1']),'respondent join')
ids <- sort(b$student_id); p <- p[match(ids,p$student_id),]; b <- b[match(ids,b$student_id),]; y <- y[match(ids,y$student_id),]
need(all(p$college_id==b$college_id),'college mismatch'); keep <- b$offer=='0'; p <- p[keep,]; b <- b[keep,]; y <- y[keep,]
need(all(y$exit_type %in% c('none','withdrawal','transfer','consent_withdrawn')),'exit type'); unknown <- sum(y$exit_type=='consent_withdrawn'); known <- y$exit_type!='consent_withdrawn'; p <- p[known,]; y <- y[known,]
need(all(y$withdrawn_12==as.character(as.integer(y$exit_type=='withdrawal'))),'label mismatch')
need(all(p$part_time %in% c('0','1')) && all(p$program_area %in% c('Arts and Humanities','Social Sciences','Business','STEM','Health')),'entry category'); grade <- as.numeric(p$prior_grade);need(all(is.finite(grade)&grade>=0&grade<=4),'entry grade')
people <- data.frame(row_id=p$student_id,college_id=p$college_id,part_time=as.integer(p$part_time),grade=grade,category=paste(p$college_id,p$program_area,p$part_time,sep='|'),y=as.integer(y$withdrawn_12),exit_type=y$exit_type,stringsAsFactors=FALSE)
outer <- read(file.path(root,'splits/unit05_outer.csv')); inner <- read(file.path(root,'splits/unit05_inner.csv'))
need(nrow(outer)==nrow(people) && !anyDuplicated(outer$row_id) && setequal(outer$row_id,people$row_id),'outer IDs');outer <- outer[match(people$row_id,outer$row_id),]
need(all(outer$college_id==people$college_id) && all(outer$outer_fold %in% as.character(1:5)) && setequal(outer$outer_fold,as.character(1:5)),'outer college/fold')
need(all(inner$outer_fold %in% as.character(1:5)),'unexpected inner outer fold');nested <- list()
for(f in as.character(1:5)) {
 t <- people[outer$outer_fold!=f,]; z <- inner[inner$outer_fold==f,]
 need(nrow(z)==nrow(t) && !anyDuplicated(z$row_id) && setequal(z$row_id,t$row_id),'inner membership'); z <- z[match(t$row_id,z$row_id),]
 need(all(z$college_id==t$college_id) && all(z$inner_fold %in% as.character(1:5)) && setequal(z$inner_fold,as.character(1:5)),'inner college/fold');nested[[f]] <- setNames(z$inner_fold,z$row_id)
}
# Read the compact specification without third-party JSON packages. The fixed
# procedure is audited against the complete JSON by the Python source gate.
spec <- paste(readLines(file.path(root,'model-specs/unit05.json')),collapse='\n')
need(grepl('"smoothing_weight": 5',spec,fixed=TRUE),'model spec')
allow <- regmatches(spec,regexpr('"predictor_allowlist"[[:space:]]*:[[:space:]]*\\[[^]]*\\]',spec,perl=TRUE))
tokens <- regmatches(allow,gregexpr('"[^"]+"',allow))[[1]]
need(identical(gsub('"','',tokens[-1],fixed=TRUE),c('college_id','program_area','part_time','prior_grade')),'predictor allowlist')
status <- c(prevalence='valid',risk_table='valid',encoding_nested='valid',encoding_in_sample='training_mismatch_outer_evaluation_honest',encoding_pooled='invalid_validation_labels_in_encoding',table_pooled_cutpoints='invalid_validation_features_in_transform',future_registry='invalid_future_outcome_predictor')
outputs <- setNames(lapply(1:9,function(i)list()),c('predictions','coefficients','encodings','training_features','fit_rows','cells','transforms','scores','summary'))
add <- function(name,row) outputs[[name]][[length(outputs[[name]])+1L]] <<- row
encoder <- function(t) {
 need(nrow(t)>0,'empty encoding fit');prior <- sum(t$y)/nrow(t);keys <- sort(unique(t$category));counts <- lapply(keys,function(k){yy <- t$y[t$category==k];c(n=length(yy),events=sum(yy),encoding=(sum(yy)+5*prior)/(length(yy)+5))});names(counts) <- keys; list(prior=prior,counts=counts)
}
encoded <- function(row,e) {key <- row$category; if(key %in% names(e$counts)) e$counts[[key]]['encoding'] else e$prior}
recorded <- function(f,stage,j,t) {
 e <- encoder(t)
 for(i in seq_len(nrow(t))) add('fit_rows',data.frame(fold=f,stage=stage,inner_fold=j,row_id=t$row_id[i]))
 for(k in sort(names(e$counts))) {v <- e$counts[[k]];add('encodings',data.frame(fold=f,stage=stage,inner_fold=j,category=k,n=unname(v['n']),events=unname(v['events']),prevalence=e$prior,encoding=unname(v['encoding'])))}
 e
}
linear <- function(x,y) {
 spread <- sum((x-mean(x))^2);need(spread>1e-12,'encoding rank');fit <- lm.fit(cbind(1,x),y,tol=1e-12);need(fit$rank==2L && all(is.finite(fit$coefficients)),'encoding rank');c(intercept=fit$coefficients[1],slope=fit$coefficients[2],spread=spread)
}
score <- function(y,p) {
 need(length(y)>0 && all(is.finite(p)&p>=0&p<=1),'score probabilities');a <- p[y==1];b <- p[y==0];pairs <- length(a)*length(b);twice <- sum(vapply(a,function(q)2*sum(q>b)+sum(q==b),numeric(1)));data.frame(n=length(y),events=sum(y),brier=sum((y-p)^2)/length(y),auc=if(pairs>0)twice/(2*pairs) else '',auc_pairs=pairs,auc_twice_wins=twice)
}
oracle <- score(c(1,0,1,0),c(.8,.2,.5,.5));need(oracle$auc_pairs==4 && oracle$auc_twice_wins==7 && abs(oracle$auc-.875)<1e-14 && abs(oracle$brier-.145)<1e-14,'score oracle')
line_oracle <- linear(c(.1,.3,.5),c(.2,.6,1));need(abs(line_oracle[1])<1e-12 && abs(line_oracle[2]-2)<1e-12,'fit oracle')
for(f in as.character(1:5)) {
 train <- people[outer$outer_fold!=f,]; validation <- people[outer$outer_fold==f,];e <- recorded(f,'outer_encoder','',train);bad <- recorded(f,'invalid_pooled_encoder','',people);crossfit <- setNames(rep(NA_real_,nrow(train)),train$row_id)
 for(i in seq_len(nrow(people))) add('fit_rows',data.frame(fold=f,stage='invalid_pooled_cutpoints',inner_fold='',row_id=people$row_id[i]))
 for(j in as.character(1:5)) {
  z <- nested[[f]][train$row_id]; permitted <- train[z!=j,];ie <- recorded(f,'inner_encoder',j,permitted)
  for(i in which(z==j)) crossfit[train$row_id[i]] <- encoded(train[i,],ie)
 }
 need(all(is.finite(crossfit)),'crossfit coverage');parameters <- list()
 for(pipeline in c('encoding_nested','encoding_in_sample','encoding_pooled')) {
  x <- if(pipeline=='encoding_nested')unname(crossfit[train$row_id]) else vapply(seq_len(nrow(train)),function(i)encoded(train[i,],if(pipeline=='encoding_pooled')bad else e),numeric(1));beta <- linear(x,train$y);parameters[[pipeline]] <- beta
  add('coefficients',data.frame(fold=f,pipeline=pipeline,intercept=unname(beta[1]),slope=unname(beta[2]),n_fit=nrow(train),x_sst=unname(beta[3])))
  for(i in seq_len(nrow(train))) {
   add('training_features',data.frame(fold=f,pipeline=pipeline,row_id=train$row_id[i],inner_fold=if(pipeline=='encoding_nested')nested[[f]][train$row_id[i]] else '',encoding=x[i],own_label_included=as.integer(pipeline!='encoding_nested')))
   add('fit_rows',data.frame(fold=f,stage=paste0('calibration_',pipeline),inner_fold='',row_id=train$row_id[i]))
  }
 }
 tables <- list()
 for(pipeline in c('risk_table','table_pooled_cutpoints')) {
  cuts <- quantile(if(pipeline=='risk_table')train$grade else people$grade,c(.25,.5,.75),type=7,names=FALSE);risks <- numeric(8);keys <- 4*train$part_time+vapply(train$grade,function(g)sum(g>cuts),integer(1))
  for(k in 0:7) {
   yy <- train$y[keys==k];n <- length(yy);s <- sum(yy);risk <- if(n)s/n else e$prior;risks[k+1] <- risk
   add('cells',data.frame(fold=f,pipeline=pipeline,part_time=k%/%4,grade_bin=k%%4+1,n=n,events=s,risk=risk,fallback=as.integer(n==0)))
  }
  tables[[pipeline]] <- list(cuts=cuts,risks=risks)
  for(i in 1:3)add('transforms',data.frame(fold=f,pipeline=pipeline,quantile=c(.25,.5,.75)[i],value=cuts[i]))
 }
 for(pipeline in names(status))for(i in seq_len(nrow(validation))) {
  row <- validation[i,];feature <- '';fallback <- 0L;clipped <- 0L
  if(pipeline=='prevalence')probability <- e$prior else if(pipeline=='future_registry')probability <- as.numeric(row$exit_type=='withdrawal') else if(pipeline %in% names(tables)) {
   table <- tables[[pipeline]];probability <- table$risks[4*row$part_time+sum(row$grade>table$cuts)+1]
  } else {
   encoder_for_prediction <- if(pipeline=='encoding_pooled')bad else e;feature <- encoded(row,encoder_for_prediction);fallback <- as.integer(!row$category %in% names(encoder_for_prediction$counts));beta <- parameters[[pipeline]];rawp <- beta[1]+beta[2]*feature;probability <- min(1,max(0,rawp));clipped <- as.integer(rawp<0||rawp>1)
  }
  add('predictions',data.frame(fold=f,row_id=row$row_id,pipeline=pipeline,status=unname(status[pipeline]),y=row$y,prediction=unname(probability),encoding=unname(feature),fallback=fallback,clipped=clipped))
 }
}
predictions <- do.call(rbind,outputs$predictions)
for(f in c(as.character(1:5),'pooled')) {
 for(pipeline in names(status)) {
  z <- predictions[predictions$pipeline==pipeline & (f=='pooled'|predictions$fold==f),];add('scores',cbind(data.frame(fold=f,scope='eligible',pipeline=pipeline,status=unname(status[pipeline])),score(z$y,z$prediction)))
 }
 z <- predictions[predictions$pipeline=='risk_table' & (f=='pooled'|predictions$fold==f),];z <- z[people$exit_type[match(z$row_id,people$row_id)]!='transfer',];add('scores',cbind(data.frame(fold=f,scope='future_no_transfer',pipeline='risk_table',status='changed_target_invalid_enrollment_filter'),score(z$y,z$prediction)))
}
counts <- c(pool_n=nrow(people)+unknown,eligible_n=nrow(people),unknown_n=unknown,events=sum(people$y),transfer_n=sum(people$exit_type=='transfer'),colleges=length(unique(people$college_id)),outer_folds=5,inner_memberships=nrow(inner),prediction_records=nrow(predictions),fit_memberships=length(outputs$fit_rows),encoder_maps=length(outputs$encodings),calibration_fits=length(outputs$coefficients),fallback_nested=sum(predictions$fallback[predictions$pipeline=='encoding_nested']),clipped_nested=sum(predictions$clipped[predictions$pipeline=='encoding_nested']))
for(k in sort(names(counts)))add('summary',data.frame(name=k,value=unname(counts[k])))
dir.create(dest,recursive=TRUE,showWarnings=FALSE)
for(k in names(outputs))write.table(do.call(rbind,outputs[[k]]),file.path(dest,paste0(k,'.csv')),sep=',',row.names=FALSE,col.names=TRUE,quote=TRUE,na='')
cat('UNIT05_BASE_R_OK predictions=',nrow(predictions),' R=',as.character(getRversion()),'\n',sep='')
