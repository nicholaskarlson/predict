#!/usr/bin/env Rscript
# Independent base-R raw joins, constants, tables, predictions and losses.
options(stringsAsFactors=FALSE, digits=17)
invisible(Sys.setlocale('LC_COLLATE','C'))
need <- function(ok,msg) if (!isTRUE(ok)) stop(msg,call.=FALSE)
read <- function(path) read.csv(path,colClasses='character',na.strings=NULL,check.names=FALSE)
records <- function(root) {
  output <- list()
  for (cohort in c('anchor','cohort2')) {
    folder <- file.path(root,'source/tonmick-v1.0.0/data',cohort)
    tables <- lapply(c('participants','baseline','registry'),function(n) {
      x <- read(file.path(folder,paste0(n,'.csv')))
      need(!anyDuplicated(x$student_id),paste('duplicate',n));x
    })
    p <- tables[[1]];b <- tables[[2]];o <- tables[[3]]
    need(setequal(b$student_id,o$student_id) && setequal(b$student_id,p$student_id[p$responded_baseline=='1']),'respondent join')
    for (i in sort(b$student_id)) {
      person <- p[p$student_id==i,,drop=FALSE];base <- b[b$student_id==i,,drop=FALSE];out <- o[o$student_id==i,,drop=FALSE]
      need(person$college_id==base$college_id,'college mismatch')
      if (cohort=='anchor' && base$offer!='0') next
      if (cohort=='cohort2' && person$college_group!='continuing') next
      need(person$part_time %in% c('0','1'),'part-time coding')
      prior <- as.numeric(person$prior_grade);need(is.finite(prior) && prior>=0 && prior<=4,'prior grade')
      kind <- out$exit_type;need(kind %in% c('none','withdrawal','transfer','consent_withdrawn'),'exit type')
      unknown <- kind=='consent_withdrawn'
      if (!unknown) need(out$withdrawn_12==as.character(as.integer(kind=='withdrawal')) && out$transferred_12==as.character(as.integer(kind=='transfer')),'known label')
      grade <- if (out$gpa12=='') NA_real_ else as.numeric(out$gpa12)
      need(out$gpa12=='' || (is.finite(grade) && grade>=0 && grade<=4 && !unknown),'grade observation')
      need(out$gpa_at_exit %in% c('0','1'),'exit-grade flag')
      if (!is.na(grade) && out$gpa_at_exit=='1') need(kind %in% c('withdrawal','transfer') && as.numeric(out$exit_month)>=4,'exit-grade timing')
      eligible <- if (cohort=='anchor') 'train' else 'final'
      output[[length(output)+1L]] <- data.frame(row_id=i,cohort=cohort,college_id=person$college_id,part_time=as.integer(person$part_time),
        prior_grade=prior,exit_type=kind,gpa_at_exit=as.integer(out$gpa_at_exit),withdrawal=if (unknown) NA_integer_ else as.integer(kind=='withdrawal'),
        first_exit=if (unknown) NA_integer_ else as.integer(kind %in% c('withdrawal','transfer')),grade=grade,
        binary_role=if (unknown) 'excluded_unknown' else eligible,grade_role=if (is.na(grade)) 'excluded_missing_grade' else eligible)
    }
  }
  rows <- do.call(rbind,output);rows <- rows[order(rows$row_id),];rownames(rows) <- NULL
  need(!anyDuplicated(rows$row_id),'cross-cohort key')
  roles <- read(file.path(root,'splits/unit02_roles.csv'))
  need(identical(roles,rows[c('row_id','binary_role','grade_role')]),'role manifest');rows
}
fit <- function(rows) {
  binary <- rows[rows$binary_role=='train',];grade <- rows[rows$grade_role=='train',]
  need(nrow(binary)>0 && nrow(grade)>0,'empty fit')
  cuts <- as.numeric(quantile(binary$prior_grade,c(.25,.5,.75),type=7,names=FALSE))
  params <- list(cuts=cuts,mean=mean(grade$grade),median=median(grade$grade));cells <- list();fits <- list()
  for (recipe in c('mean','median')) fits[[length(fits)+1L]] <- data.frame(target='grade',recipe=recipe,component='constant',value=params[[recipe]])
  bins <- 1L+vapply(binary$prior_grade,function(x) sum(x>cuts),integer(1))
  for (target in c('first_exit','withdrawal')) {
    need(all(binary[[target]] %in% c(0,1)),'binary fit labels')
    prevalence <- mean(binary[[target]]);table <- numeric(8)
    fits[[length(fits)+1L]] <- data.frame(target=target,recipe='prevalence',component='constant',value=prevalence)
    for (j in 1:3) fits[[length(fits)+1L]] <- data.frame(target=target,recipe='risk_table',component=paste0('cut_',j),value=cuts[j])
    for (part in 0:1) for (bin in 1:4) {
      members <- binary[binary$part_time==part & bins==bin,];n <- nrow(members);events <- sum(members[[target]])
      probability <- if (n==0L) prevalence else events/n;table[4L*part+bin] <- probability
      cells[[length(cells)+1L]] <- data.frame(target=target,part_time=part,grade_bin=bin,n=n,events=events,prediction=probability,fallback=as.integer(n==0L))
    }
    params[[target]] <- list(prevalence=prevalence,table=table)
  }
  fits <- do.call(rbind,fits);fits <- fits[order(fits$target,fits$recipe,fits$component),];rownames(fits) <- NULL
  list(params=params,fits=fits,cells=do.call(rbind,cells))
}
binary_losses <- function(y,p) {
  need(length(y)>0 && length(y)==length(p) && all(y %in% c(0,1)),'binary scoring labels')
  need(all(is.finite(p)) && all(p>=0 & p<=1),'probability range')
  clipped <- pmin(pmax(p,1e-12),1-1e-12)
  list(squared=(p-y)^2,log=ifelse(y==1,-log(clipped),-log1p(-clipped)),clipped=sum(p!=clipped))
}
calculate <- function(root) {
  rows <- records(root);fitted <- fit(rows);params <- fitted$params
  summary <- list();scores <- list();predictions <- list();examples <- list();fit_rows <- list()
  for (cohort in c('anchor','cohort2')) {
    r <- rows[rows$cohort==cohort,]
    counts <- list(pool=nrow(r),known=sum(r$binary_role!='excluded_unknown'),unknown=sum(r$binary_role=='excluded_unknown'),
      withdrawals=sum(r$withdrawal==1,na.rm=TRUE),transfers=sum(r$exit_type=='transfer'),grade_observed=sum(!is.na(r$grade)),
      grade_missing=sum(is.na(r$grade)),exit_grades=sum(!is.na(r$grade) & r$gpa_at_exit==1))
    for (key in names(counts)) summary[[paste0(cohort,'_',key)]] <- counts[[key]]
  }
  for (target in c('first_exit','grade','withdrawal')) {
    rolekey <- if (target=='grade') 'grade_role' else 'binary_role'
    training <- rows[rows[[rolekey]]=='train',];final <- rows[rows[[rolekey]]=='final',];need(nrow(final)>0,'empty final')
    fit_rows[[length(fit_rows)+1L]] <- data.frame(target=target,row_id=training$row_id)
    y <- final[[target]];summary[[paste0(target,'_final_n')]] <- nrow(final)
    if (target!='grade') {
      summary[[paste0(target,'_train_prevalence')]] <- params[[target]]$prevalence
      summary[[paste0(target,'_final_prevalence')]] <- mean(y)
    }
    recipes <- if (target=='grade') c('mean','median','prior_grade') else c('prevalence','risk_table')
    for (recipe in recipes) {
      if (target=='grade') p <- if (recipe=='prior_grade') final$prior_grade else rep(params[[recipe]],nrow(final))
      else if (recipe=='prevalence') p <- rep(params[[target]]$prevalence,nrow(final))
      else {
        bins <- 1L+vapply(final$prior_grade,function(x) sum(x>params$cuts),integer(1))
        p <- params[[target]]$table[4L*final$part_time+bins]
      }
      squared <- (p-y)^2;absolute <- abs(p-y);clipping <- 0L;logs <- rep(NA_real_,length(p))
      if (target=='grade') measures <- list(MAE=mean(absolute),MSE=mean(squared))
      else {
        loss <- binary_losses(y,p);squared <- loss$squared;logs <- loss$log;clipping <- loss$clipped
        measures <- list(Brier=mean(squared),log_loss=mean(logs))
      }
      for (measure in sort(names(measures))) scores[[length(scores)+1L]] <- data.frame(target=target,recipe=recipe,n=nrow(final),measure=measure,
        value=measures[[measure]],clipped=if (measure=='log_loss') clipping else 0L)
      predictions[[length(predictions)+1L]] <- data.frame(target=target,recipe=recipe,row_id=final$row_id,y=y,prediction=p)
      j <- seq_len(min(3L,nrow(final)))
      examples[[length(examples)+1L]] <- data.frame(target=target,recipe=recipe,row_id=final$row_id[j],y=y[j],prediction=p[j],squared=squared[j],absolute=absolute[j],log_loss=logs[j])
    }
  }
  summary$grade_train_mean <- params$mean;summary$grade_train_median <- params$median
  summary$withdrawal_wrong_unknown_denominator <- summary$anchor_withdrawals/summary$anchor_pool
  summary$first_exit_wrong_unknown_denominator <- (summary$anchor_withdrawals+summary$anchor_transfers)/summary$anchor_pool
  summary$grade_wrong_zero_mean <- params$mean*summary$anchor_grade_observed/summary$anchor_pool
  names_sorted <- sort(names(summary));summary <- data.frame(name=names_sorted,value=vapply(summary[names_sorted],as.numeric,numeric(1)))
  list(summary=summary,fits=fitted$fits,cells=fitted$cells,scores=do.call(rbind,scores),predictions=do.call(rbind,predictions),examples=do.call(rbind,examples),fit_rows=do.call(rbind,fit_rows))
}
emit <- function(result,out) {
  dir.create(out,recursive=TRUE,showWarnings=FALSE)
  for (name in names(result)) {
    x <- result[[name]]
    for (column in names(x)) if (is.numeric(x[[column]])) x[[column]] <- ifelse(is.na(x[[column]]),'',sprintf('%.17g',x[[column]]))
    write.table(x,file.path(out,paste0(name,'.csv')),sep=',',quote=FALSE,row.names=FALSE,na='',col.names=TRUE)
  }
}
args <- commandArgs(trailingOnly=TRUE)
if (identical(args,'--self-test')) {
  x <- binary_losses(c(0,1),c(.25,.75));need(all(x$squared==1/16),'Brier hand oracle')
  need(max(abs(x$log+log(.75)))<1e-15 && x$clipped==0,'log hand oracle')
  x <- binary_losses(c(0,1),c(0,1));need(x$clipped==2 && all(is.finite(x$log)),'endpoint clipping')
  need(inherits(try(binary_losses(c(0,.5),c(.2,.2)),silent=TRUE),'try-error'),'nonbinary label rejection')
  need(inherits(try(binary_losses(c(0,1),c(-.1,.2)),silent=TRUE),'try-error'),'invalid probability rejection')
  need(mean((c(0,0,3)-1)^2)==2 && mean(abs(c(0,0,3)-0))==1,'grade loss oracle')
  need(as.numeric(quantile(c(1,2,3,4),.25,type=7))==1.75,'type7 interpolation')
  cat('UNIT02_R_ORACLES_OK losses,clipping,labels,type7\n')
} else {
  need(length(args)==2,'usage: unit02_base_r.R RESOURCE OUTDIR')
  result <- calculate(args[1]);emit(result,args[2]);cat('UNIT02_BASE_R_OK predictions=',nrow(result$predictions),' ',R.version.string,'\n',sep='')
}
