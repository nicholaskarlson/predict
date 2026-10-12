# Independent observed joins, fixed-college slots, LINPACK QR, AIC and scores.
# No Python analyzer or committed predictions are read. Frozen JSON is checked
# by the Python audit orchestrator; this route separately enforces its design.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L) stop('usage: Rscript unit03_base_r.R RESOURCE OUTDIR')
root <- args[1]; out <- args[2]; options(digits=17); Sys.setlocale('LC_COLLATE','C')
read_chars <- function(p) read.csv(p,colClasses='character',na.strings=NULL,check.names=FALSE)
raw <- lapply(c('participants','baseline','registry'),function(k) {
 d <- read_chars(file.path(root,'source/tonmick-v1.0.0/data/anchor',paste0(k,'.csv')))
 if (anyDuplicated(d$student_id)) stop('duplicate raw key'); d
})
f <- raw[[1]]; b <- raw[[2]]; g <- raw[[3]]
stopifnot(setequal(b$student_id,g$student_id),setequal(b$student_id,f$student_id[f$responded_baseline=='1']))
f <- f[match(b$student_id,f$student_id),,drop=FALSE]; g <- g[match(b$student_id,g$student_id),,drop=FALSE]
stopifnot(all(f$student_id==b$student_id),all(g$student_id==b$student_id),all(f$college_id==b$college_id))
keep <- b$offer=='0'; f <- f[keep,,drop=FALSE]; g <- g[keep,,drop=FALSE]
observed <- g$gpa12!=''; label <- rep(NA_real_,nrow(g));label[observed] <- as.numeric(g$gpa12[observed])
prior <- as.numeric(f$prior_grade);age <- as.numeric(f$age)
stopifnot(all(is.finite(label[observed])),all(label[observed]>=0 & label[observed]<=4),
 all(!observed[g$exit_type=='consent_withdrawn']),all(f$part_time %in% c('0','1')),all(f$first_gen %in% c('0','1')),
 all(is.finite(prior)),all(prior>=0 & prior<=4),all(is.finite(age)),all(age>=16 & age<=80))
rows <- data.frame(row_id=f$student_id,college_id=f$college_id,region=f$region,part_time=as.integer(f$part_time),first_gen=as.integer(f$first_gen),age=age,prior_grade=prior,exit_type=g$exit_type,gpa_at_exit=as.integer(g$gpa_at_exit),y=label,
 role=ifelse(!observed,'excluded_missing_grade',ifelse(f$region=='West','development','setting_check')),stringsAsFactors=FALSE)
rows <- rows[order(rows$row_id),,drop=FALSE]; roles <- read_chars(file.path(root,'splits/unit03_roles.csv'))
for (key in c('row_id','college_id','region','role')) stopifnot(identical(as.character(rows[[key]]),roles[[key]]))
dev <- rows[rows$role=='development',,drop=FALSE]; setting <- rows[rows$role=='setting_check',,drop=FALSE]
draws <- read_chars(file.path(root,'splits/unit03_bootstrap.csv')); jobs <- list(original=dev);offset <- 0L
for (number in 1:200) {
 rep <- sprintf('B%04d',number);parts <- list()
 for (college in sort(unique(dev$college_id))) {
  n <- sum(dev$college_id==college);a <- draws[seq.int(offset+1L,offset+n),,drop=FALSE];offset <- offset+n
  stopifnot(all(a$replicate==rep),all(a$college_id==college),identical(a$slot,sprintf('%03d',seq_len(n))))
  ix <- match(a$row_id,dev$row_id);stopifnot(!anyNA(ix),all(dev$college_id[ix]==college))
  d <- dev[ix,,drop=FALSE];d$slot <- paste0(college,'-',a$slot);parts[[college]] <- d
 }
 jobs[[rep]] <- do.call(rbind,parts)
}
stopifnot(offset==nrow(draws))
models <- c('entry','baseline','rich')
matrix_for <- function(d,m,params) {
 z <- (d$prior_grade-params$prior_grade[1])/params$prior_grade[2]
 a <- (d$age-params$age[1])/params$age[2]
 x <- cbind(intercept=rep(1,nrow(d)),z_prior_grade=z)
 if (m!='entry') x <- cbind(x,part_time=d$part_time,first_gen=d$first_gen,z_age=a)
 if (m=='rich') x <- cbind(x,z_prior_grade_squared=z*z,z_age_squared=a*a,z_prior_grade_part_time=z*d$part_time,z_prior_grade_first_gen=z*d$first_gen,z_prior_grade_z_age=z*a)
 x
}
qr_fit <- function(x,y) {
 stopifnot(nrow(x)>=ncol(x));decomp <- qr(x,tol=1e-10,LAPACK=FALSE)
 stopifnot(decomp$rank==ncol(x),identical(decomp$pivot,seq_len(ncol(x))))
 triangular <- qr.R(decomp);diagonal <- abs(diag(triangular));stopifnot(all(diagonal>1e-10*max(diagonal)))
 condition <- norm(triangular,'I')*norm(backsolve(triangular,diag(ncol(x))),'I')
 stopifnot(is.finite(condition),condition<=1e5)
 beta <- as.numeric(qr.coef(decomp,y));e <- y-as.numeric(x %*% beta)
 gradient <- max(abs(as.numeric(crossprod(x,e))))/length(y)
 stopifnot(all(is.finite(beta)),gradient<=1e-10,mean(e^2)>0)
 list(beta=beta,condition=condition,score=gradient,mse=mean(e^2))
}
metrics <- function(d,p) {
 variance <- mean((d$y-mean(d$y))^2);stopifnot(variance>1e-12)
 mse <- mean((d$y-p)^2);c(mse=mse,r2=1-mse/variance)
}
select <- function(values) models[which(values-min(values)<=1e-10)[1]]
# A known-line oracle checks the solver separately from Tonmick.
oracle <- qr_fit(cbind(1,c(-1,0,1,2)),c(-1,2,5,8)+c(1e-6,-1e-6,-1e-6,1e-6))
stopifnot(max(abs(oracle$beta-c(2,3)))<1e-10)
outputs <- setNames(lapply(1:10,function(i) list()),c('coefficients','transforms','fit_rows','diagnostics','scores','selections','optimism','corrections','predictions','examples'))
add <- function(name,d) outputs[[name]][[length(outputs[[name]])+1L]] <<- d
original_choice <- ''
for (rep in names(jobs)) {
 train <- jobs[[rep]]; params <- lapply(c('prior_grade','age'),function(key) {
  mu <- mean(train[[key]]);sd <- sqrt(mean((train[[key]]-mu)^2));stopifnot(is.finite(sd),sd>1e-12);c(mu,sd)
 });names(params) <- c('prior_grade','age')
 for (key in names(params)) add('transforms',data.frame(replicate=rep,column=key,mean=params[[key]][1],sd=params[[key]][2]))
 slots <- if (rep=='original') sprintf('%03d',seq_len(nrow(train))) else train$slot
 add('fit_rows',data.frame(replicate=rep,slot=slots,row_id=train$row_id,college_id=train$college_id))
 fitted <- list();aics <- c()
 for (model in models) {
  x <- matrix_for(train,model,params);fit <- qr_fit(x,train$y);aic <- nrow(train)*log(fit$mse)+2*(ncol(x)+1L)
  fitted[[model]] <- fit$beta;aics[model] <- aic
  add('diagnostics',data.frame(replicate=rep,model_id=model,n_fit=nrow(train),p=ncol(x),rank=ncol(x),condition_r_inf=fit$condition,score_inf=fit$score,fit_mse=fit$mse,aic=aic))
  add('coefficients',data.frame(replicate=rep,model_id=model,term=colnames(x),coefficient=fit$beta))
 }
 chosen <- select(aics);if (rep=='original') original_choice <- chosen
 add('selections',data.frame(replicate=rep,selected_model=chosen,selected_aic=unname(aics[chosen])))
 scopes <- if (rep=='original') list(apparent=dev,setting_check=setting) else list(bootstrap=train,original_West=dev)
 scores <- list();forecasts <- list()
 for (model in models) for (scope in names(scopes)) {
  d <- scopes[[scope]];p <- as.numeric(matrix_for(d,model,params) %*% fitted[[model]])
  key <- paste(model,scope,sep=':');scores[[key]] <- metrics(d,p);forecasts[[key]] <- p
  add('scores',data.frame(replicate=rep,scope=scope,model_id=model,n=nrow(d),mse=scores[[key]]['mse'],r2=scores[[key]]['r2']))
 }
 workflows <- if (rep=='original') setNames(models,paste0('candidate_',models)) else c(aic_selection=chosen,entry_fixed='entry')
 for (workflow in names(workflows)) for (scope in names(scopes)) {
  model <- workflows[[workflow]];d <- scopes[[scope]];p <- forecasts[[paste(model,scope,sep=':')]]
  slots <- if ('slot' %in% names(d)) d$slot else sprintf('%03d',seq_len(nrow(d)))
  add('predictions',data.frame(replicate=rep,scope=scope,workflow=workflow,model_id=model,slot=slots,row_id=d$row_id,y=d$y,prediction=p,squared_error=(d$y-p)^2))
 }
 if (rep!='original') for (workflow in c('aic_selection','entry_fixed','frozen_choice_incomplete')) {
  model <- switch(workflow,aic_selection=chosen,entry_fixed='entry',frozen_choice_incomplete=original_choice)
  boot <- scores[[paste(model,'bootstrap',sep=':')]];orig <- scores[[paste(model,'original_West',sep=':')]]
  add('optimism',data.frame(replicate=rep,workflow=workflow,model_id=model,bootstrap_mse=boot['mse'],original_mse=orig['mse'],mse_gap=orig['mse']-boot['mse'],bootstrap_r2=boot['r2'],original_r2=orig['r2'],r2_gap=boot['r2']-orig['r2']))
 }
}
for (name in names(outputs)[!names(outputs) %in% c('corrections','examples')]) outputs[[name]] <- do.call(rbind,outputs[[name]])
for (workflow in c('aic_selection','entry_fixed','frozen_choice_incomplete')) {
 model <- if (workflow=='entry_fixed') 'entry' else original_choice
 a <- outputs$scores[outputs$scores$replicate=='original' & outputs$scores$scope=='apparent' & outputs$scores$model_id==model,,drop=FALSE]
 d <- outputs$optimism[outputs$optimism$workflow==workflow,,drop=FALSE]
 add('corrections',data.frame(workflow=workflow,original_model=model,replicates=nrow(d),apparent_mse=a$mse,mean_mse_gap=mean(d$mse_gap),corrected_mse=a$mse+mean(d$mse_gap),mse_gap_mcse=sd(d$mse_gap)/sqrt(nrow(d)),apparent_r2=a$r2,mean_r2_gap=mean(d$r2_gap),corrected_r2=a$r2-mean(d$r2_gap),r2_gap_mcse=sd(d$r2_gap)/sqrt(nrow(d))))
}
outputs$corrections <- do.call(rbind,outputs$corrections)
summary <- c()
for (group in c('West','other')) {
 d <- rows[if (group=='West') rows$region=='West' else rows$region!='West',,drop=FALSE];v <- d[!is.na(d$y),,drop=FALSE]
 values <- c(pool_n=nrow(d),grade_n=nrow(v),missing_n=nrow(d)-nrow(v),college_n=length(unique(v$college_id)),exit_grade_n=sum(v$gpa_at_exit))
 names(values) <- paste0(group,'_',names(values));summary <- c(summary,values)
}
for (model in models) summary[paste0('selected_',model,'_n')] <- sum(outputs$selections$replicate!='original' & outputs$selections$selected_model==model)
summary <- c(summary,replicates_n=200,fits_n=nrow(outputs$diagnostics),fit_slots_n=nrow(outputs$fit_rows),predictions_n=nrow(outputs$predictions))
outputs$summary <- data.frame(name=sort(names(summary)),value=unname(summary[sort(names(summary))]))
d <- outputs$predictions[outputs$predictions$replicate=='B0001' & outputs$predictions$workflow=='aic_selection' & outputs$predictions$scope=='original_West',,drop=FALSE]
outputs$examples <- head(d,3)
dir.create(out,recursive=TRUE,showWarnings=FALSE)
keys <- c('replicate','scope','workflow','model_id','slot','row_id','column','term','name')
for (name in names(outputs)) {
 d <- outputs[[name]];sortkeys <- intersect(names(d),keys)
 if (length(sortkeys)) d <- d[do.call(order,lapply(d[sortkeys],as.character)),,drop=FALSE]
 for (key in names(d)) if (is.numeric(d[[key]])) d[[key]] <- vapply(d[[key]],function(v) sprintf('%.17g',v),character(1))
 write.table(d,file.path(out,paste0(name,'.csv')),sep=',',row.names=FALSE,col.names=TRUE,quote=FALSE,na='')
}
writeLines(capture.output(sessionInfo()),file.path(out,'session-info.txt'))
cat('UNIT03_BASE_R_OK fits=',nrow(outputs$diagnostics),' predictions=',nrow(outputs$predictions),' R=',as.character(getRversion()),' solver=LINPACK_QR\n',sep='')
