# Independent base R: explicit design columns, rank-sum AUC and aggregate arms.
read_unit12 <- function(root, name) read.csv(file.path(root,name),colClasses='character',stringsAsFactors=FALSE,check.names=FALSE,na.strings=character())
unit12_terms <- c('intercept','part_time','first_gen','z_age','gender_male','gender_another','program_Social_Sciences','program_STEM','program_Health','program_Business','z_prior_grade')
unit12_roles <- c('temporal_continuing','temporal_new')
frame_matrix <- function(d,t) {
  stopifnot(all(d$gender %in% c('female','male','another')),all(d$program_area %in% c('Arts and Humanities','Social Sciences','STEM','Health','Business')),
    all(d$part_time %in% c('0','1')),all(d$first_gen %in% c('0','1')))
  age <- as.numeric(d$age);grade <- as.numeric(d$prior_grade)
  stopifnot(all(is.finite(age)),all(is.finite(grade)),all(age>=16 & age<=80),all(grade>=0 & grade<=4))
  cbind(1,as.numeric(d$part_time),as.numeric(d$first_gen),(age-t[['age']][1])/t[['age']][2],
    as.numeric(d$gender=='male'),as.numeric(d$gender=='another'),as.numeric(d$program_area=='Social Sciences'),
    as.numeric(d$program_area=='STEM'),as.numeric(d$program_area=='Health'),as.numeric(d$program_area=='Business'),
    (grade-t[['prior_grade']][1])/t[['prior_grade']][2])
}
unit12_score <- function(y,p) {
  stopifnot(length(y)>0,length(y)==length(p),all(y %in% c(0,1)),all(is.finite(p)),all(p>=0 & p<=1))
  n <- length(y);e <- sum(y);ne <- n-e;den <- 2*e*ne
  pairs <- if(den>0) 2*(sum(rank(p,ties.method='average')[y==1])-e*(e+1)/2) else 0
  data.frame(n=n,events=e,nonevents=ne,brier=mean((p-y)^2),mean_probability=mean(p),auc_pairs2=pairs,auc_denominator=den,
    auc=if(den>0) as.character(pairs/den) else '',auc_reason=if(den>0) '' else 'one_class',stringsAsFactors=FALSE)
}
offer_tables <- function(d) {
  stopifnot(!anyDuplicated(d$row_id),all(d$offer %in% c('0','1')),all(d$y %in% c('','0','1')),
    all(d$outcome_known==as.character(as.integer(nzchar(d$y)))))
  colleges <- sort(unique(d$college_id));total <- nrow(d);output <- list()
  for(college in c(colleges,'pooled')) {
    a <- if(college=='pooled') d else d[d$college_id==college,,drop=FALSE]
    rec <- data.frame(college_id=college,randomized_n=nrow(a),weight=if(college=='pooled') 1 else nrow(a)/total,stringsAsFactors=FALSE)
    assigned <- numeric(2)
    for(z in 0:1) {
      g <- a[a$offer==as.character(z),,drop=FALSE];n <- nrow(g);known <- g[nzchar(g$y),,drop=FALSE];k <- nrow(known);events <- sum(as.numeric(known$y))
      stopifnot(n>0,k>0);assigned[z+1] <- n
      rec[[paste0('assigned',z)]] <- n;rec[[paste0('known',z)]] <- k;rec[[paste0('unknown',z)]] <- n-k;rec[[paste0('events',z)]] <- events
      rec[[paste0('risk_known',z)]] <- events/k;rec[[paste0('risk_lower',z)]] <- events/n;rec[[paste0('risk_upper',z)]] <- (events+n-k)/n
    }
    if(college!='pooled') stopifnot(abs(assigned[1]-assigned[2])<=1)
    rec$rd_known <- rec$risk_known1-rec$risk_known0;rec$rd_lower <- rec$risk_lower1-rec$risk_upper0;rec$rd_upper <- rec$risk_upper1-rec$risk_lower0
    output[[length(output)+1]] <- rec
  }
  counts <- do.call(rbind,output);summary <- list()
  for(name in c('standardized','pooled')) {
    groups <- counts[if(name=='standardized') counts$college_id!='pooled' else counts$college_id=='pooled',,drop=FALSE]
    summary[[length(summary)+1]] <- data.frame(contrast_id=name,randomized_n=total,known_n=sum(nzchar(d$y)),unknown_n=sum(!nzchar(d$y)),
      rd_known=sum(groups$weight*groups$rd_known),rd_lower=sum(groups$weight*groups$rd_lower),rd_upper=sum(groups$weight*groups$rd_upper),stringsAsFactors=FALSE)
  }
  list(offer_counts=counts,offer_contrasts=do.call(rbind,summary))
}
compute_unit12 <- function(root) {
  d <- read_unit12(root,'data/unit12_rows.csv');causal <- read_unit12(root,'data/unit12_randomized.csv');maps <- read_unit12(root,'splits/unit12_permutations.csv')
  coefficients <- read_unit12(root,'model-specs/unit12_coefficients.csv');beta <- as.numeric(coefficients$coefficient)
  stopifnot(identical(coefficients$term,unit12_terms),all(is.finite(beta)))
  tr <- read_unit12(root,'model-specs/unit12_transforms.csv');stopifnot(setequal(tr$feature,c('age','prior_grade')),!anyDuplicated(tr$feature))
  t <- setNames(lapply(seq_len(nrow(tr)),function(i)c(as.numeric(tr$mean[i]),as.numeric(tr$sd[i]))),tr$feature)
  stopifnot(all(vapply(t,function(v)all(is.finite(v)) && v[2]>0,logical(1))))
  stopifnot(!anyDuplicated(d$row_id),all(d$role %in% c('development',unit12_roles)),all(d$y %in% c('0','1')))
  dev <- d[d$role=='development',,drop=FALSE];constant <- sum(as.numeric(dev$y))/nrow(dev)
  predictions <- list();scores <- list();draws <- list();summaries <- list()
  for(role in unit12_roles) {
    a <- d[d$role==role,,drop=FALSE];a <- a[order(a$row_id),,drop=FALSE];X <- frame_matrix(a,t);y <- as.numeric(a$y);p <- plogis(as.vector(X %*% beta))
    for(method in c('frame_only','training_prevalence')) {
      ps <- if(method=='frame_only') p else rep(constant,nrow(a))
      predictions[[length(predictions)+1]] <- data.frame(role=role,method_id=method,row_id=a$row_id,college_id=a$college_id,probability=ps,stringsAsFactors=FALSE)
      scores[[length(scores)+1]] <- cbind(data.frame(role=role,method_id=method,stringsAsFactors=FALSE),unit12_score(y,ps))
    }
    baseline <- mean((p-y)^2);deltas <- numeric(64)
    for(draw in 1:64) {
      m <- maps[maps$role==role & maps$draw_id==as.character(draw),,drop=FALSE]
      stopifnot(nrow(m)==nrow(a),!anyDuplicated(m$row_id),!anyDuplicated(m$donor_id),setequal(m$row_id,a$row_id),setequal(m$donor_id,a$row_id))
      target <- match(m$row_id,a$row_id);donor <- match(m$donor_id,a$row_id)
      stopifnot(all(m$college_id==a$college_id[target]),all(m$college_id==a$college_id[donor]))
      source <- match(m$donor_id[match(a$row_id,m$row_id)],a$row_id);changed <- X;changed[,11] <- X[source,11]
      loss <- mean((plogis(as.vector(changed %*% beta))-y)^2);deltas[draw] <- loss-baseline
      draws[[length(draws)+1]] <- data.frame(role=role,draw_id=draw,n=nrow(a),baseline_brier=baseline,permuted_brier=loss,difference=deltas[draw],stringsAsFactors=FALSE)
    }
    qs <- as.numeric(quantile(deltas,c(.025,.975),type=7))
    summaries[[length(summaries)+1]] <- data.frame(role=role,draws=64,mean_difference=mean(deltas),permutation_sd=sd(deltas),q025=qs[1],q975=qs[2],min_difference=min(deltas),max_difference=max(deltas),positive_draws=sum(deltas>0),stringsAsFactors=FALSE)
  }
  stopifnot(nrow(maps)==64*sum(d$role %in% unit12_roles))
  new <- read_unit12(root,'model-specs/unit12_new_student.csv');stopifnot(nrow(new)==1)
  eta <- as.numeric(frame_matrix(new,t) %*% beta);arms <- offer_tables(causal)
  c(list(predictions=do.call(rbind,predictions),scores=do.call(rbind,scores),permutation_draws=do.call(rbind,draws),permutation_summary=do.call(rbind,summaries)),arms,
    list(new_prediction=data.frame(record_id=new$record_id,method_id='frame_only',eta=eta,probability=plogis(eta),stringsAsFactors=FALSE),training=data.frame(n=nrow(dev),events=sum(as.numeric(dev$y)),prevalence=constant)))
}
write_unit12 <- function(out,result) {
  dir.create(out,recursive=TRUE,showWarnings=FALSE);options(digits=17)
  for(name in names(result)) write.table(result[[name]],file.path(out,paste0(name,'.csv')),sep=',',quote=FALSE,row.names=FALSE,na='',eol='\n')
}
if(sys.nframe()==0) {
  args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==2);write_unit12(args[2],compute_unit12(args[1]));cat('UNIT12_BASE_R_OK R=',as.character(getRversion()),' frozen_replay=true\n',sep='')
}
