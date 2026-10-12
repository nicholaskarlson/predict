# Independent base-R enrollment prediction and sealed scoring. No JSON package,
# no Python output input, no model fitting or World generation.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2)
resource <- args[1]; out <- args[2]
read <- function(name) read.csv(file.path(resource,name),stringsAsFactors=FALSE,check.names=FALSE,colClasses="character",na.strings=NULL)
raw <- read("source/sealed-envelope/deployment_c2.csv")
constants <- read("model-specs/sealed_envelope_constants.csv")
constant <- setNames(as.numeric(constants$value),constants$key)
groups <- read("model-specs/sealed_envelope_groups.csv")
colleges <- read("source/sealed-envelope/colleges.csv")
header <- c("student_id","college_id","region","part_time","age","gender","first_gen","program_area","prior_grade","risk12_no_transfer","exit_month","exit_type","withdrawn_12","transferred_12","gpa12","gpa_at_exit")
stopifnot(identical(names(raw),header),nrow(raw)==constant["expected_rows"],!anyDuplicated(raw$student_id),all(startsWith(raw$student_id,"SD")))
stopifnot(!anyDuplicated(colleges$college_id),!anyDuplicated(groups$college_id),setequal(groups$college_id,colleges$college_id))
stopifnot(all(raw$college_id %in% colleges$college_id),all(raw$region==colleges$region[match(raw$college_id,colleges$college_id)]))
stopifnot(all(raw$part_time %in% c("0","1")),all(raw$first_gen %in% c("0","1")),all(raw$gender %in% c("female","male","another")))
programs <- c("Arts and Humanities","Social Sciences","STEM","Health","Business")
stopifnot(all(raw$program_area %in% programs),all(raw$exit_type %in% c("none","withdrawal","transfer")))
stopifnot(all(raw$withdrawn_12==as.character(as.integer(raw$exit_type=="withdrawal"))),all(raw$transferred_12==as.character(as.integer(raw$exit_type=="transfer"))))
for (name in c("age","prior_grade","risk12_no_transfer","exit_month")) {
  values <- as.numeric(raw[[name]]); stopifnot(all(is.finite(values)))
  limits <- switch(name,age=c(16,80),prior_grade=c(0,4),risk12_no_transfer=c(0,1),exit_month=c(0,12))
  stopifnot(all(values>=limits[1]),all(values<=limits[2]))
}
coefs <- read("model-specs/unit12_coefficients.csv"); transforms <- read("model-specs/unit12_transforms.csv")
terms <- c("intercept","part_time","first_gen","z_age","gender_male","gender_another","program_Social_Sciences","program_STEM","program_Health","program_Business","z_prior_grade")
stopifnot(identical(coefs$term,terms),identical(transforms$feature,c("age","prior_grade")))
standardize <- function(name) {
  t <- transforms[transforms$feature==name,]; (as.numeric(raw[[name]])-as.numeric(t$mean))/as.numeric(t$sd)
}
design <- cbind(1,as.numeric(raw$part_time),as.numeric(raw$first_gen),standardize("age"),as.integer(raw$gender=="male"),as.integer(raw$gender=="another"),
                as.integer(raw$program_area=="Social Sciences"),as.integer(raw$program_area=="STEM"),as.integer(raw$program_area=="Health"),as.integer(raw$program_area=="Business"),standardize("prior_grade"))
eta <- drop(design %*% as.numeric(coefs$coefficient)); p <- plogis(eta)
comparator <- constant["comparator_events"]/constant["comparator_n"]
y <- as.integer(raw$withdrawn_12); q <- as.numeric(raw$risk12_no_transfer)
population <- groups$population[match(raw$college_id,groups$college_id)]
methods <- list(frame_only=p,development_prevalence=rep(comparator,nrow(raw)))

auc <- function(p,y) {
  events <- sum(y); negatives <- length(y)-events; denominator <- 2*events*negatives
  if (!denominator) return(list(pairs=0,denominator=0,value="",reason=if(length(y)) "one_class" else "empty"))
  order <- order(p); ps <- p[order]; ys <- y[order]; runs <- rle(ps)
  ends <- cumsum(runs$lengths); starts <- ends-runs$lengths+1
  total <- c(0,cumsum(ys)); yes <- total[ends+1]-total[starts]; no <- runs$lengths-yes
  previous <- cumsum(no)-no; pairs <- sum(yes*(2*previous+no))
  list(pairs=pairs,denominator=denominator,value=pairs/denominator,reason="")
}
score <- function(p,y,q) {
  stopifnot(length(p)==length(y),length(q)==length(y),all(is.finite(p)),all(p>=0 & p<=1),all(is.finite(q)),all(q>=0 & q<=1),all(y %in% c(0,1)))
  a <- auc(p,y); n <- length(y)
  if (!n) return(c(0,0,0,"","","","",a$pairs,a$denominator,a$value,a$reason,0,"","","","","empty"))
  clipped <- pmin(1-constant["clip_epsilon"],pmax(constant["clip_epsilon"],p)); gap <- p-q
  c(n,sum(y),n-sum(y),mean(y),mean(p),mean((p-y)^2),mean(-y*log(clipped)-(1-y)*log1p(-clipped)),
    a$pairs,a$denominator,a$value,a$reason,sum(p!=clipped),mean(q),mean(gap),mean(abs(gap)),sqrt(mean(gap^2)),"different_target_no_transfer_reference")
}
# Numeric values retain 17 significant digits even inside mixed CSV rows.
score_cells <- function(p,y,q) {
  v <- score(p,y,q)
  numeric_indices <- c(1:10,12:16)
  for(i in numeric_indices) if(nzchar(v[i])) v[i] <- sprintf("%.17g",as.numeric(v[i]))
  v
}
# Independent small oracles, including ties, undefined AUC and invalid inputs.
stopifnot(auc(c(.2,.2,.8,.8),c(0,1,0,1))$pairs==4,auc(c(.2,.2,.8,.8),c(0,1,0,1))$value==.5)
stopifnot(auc(c(.1,.9),c(0,0))$reason=="one_class",auc(numeric(0),integer(0))$reason=="empty")
oracle <- score(c(.2,.6),c(0,1),c(.1,.9))
stopifnot(abs(as.numeric(oracle[6])-.1)<1e-14,abs(as.numeric(oracle[14])+.1)<1e-14,abs(as.numeric(oracle[15])-.2)<1e-14,abs(as.numeric(oracle[16])-sqrt(.05))<1e-14)
stopifnot(inherits(try(score(c(1.1),c(0),c(.1)),silent=TRUE),"try-error"),inherits(try(score(c(.2),c(0),c(NaN)),silent=TRUE),"try-error"))
cat("SEALED_BASE_R_ORACLES_OK ties=true undefined=true losses=true different_target=true invalid_inputs_rejected=true\n")
scores <- list(); bins <- list()
for (group in c("all","continuing","previously_new","unseen")) {
  mask <- if(group=="all") rep(TRUE,length(y)) else population==group
  for (method in names(methods)) {
    values <- methods[[method]]
    scores[[length(scores)+1]] <- c(group,"all",method,score_cells(values[mask],y[mask],q[mask]))
    for(b in 1:5) {
      lo <- (b-1)/5; hi <- b/5; select <- mask & values>=lo & if(b==5) values<=hi else values<hi
      n <- sum(select)
      nums <- c(b,lo,hi,n,sum(y[select]),if(n) mean(values[select]) else NA,if(n) mean(y[select]) else NA,if(n) mean(q[select]) else NA)
      cells <- ifelse(is.na(nums),"",sprintf("%.17g",nums))
      bins[[length(bins)+1]] <- c(group,method,cells,if(n) "" else "empty_bin")
    }
  }
}
for (cid in sort(colleges$college_id)) {
  mask <- raw$college_id==cid; group <- groups$population[match(cid,groups$college_id)]
  for(method in names(methods))scores[[length(scores)+1]] <- c(group,cid,method,score_cells(methods[[method]][mask],y[mask],q[mask]))
}
prior <- read("expected-results/unit11_scores.csv"); comparisons <- list()
for(group in c("continuing","previously_new")) {
  role <- if(group=="continuing") "temporal_continuing" else "temporal_new"
  old <- prior[prior$method_id=="frame_only" & prior$pattern=="all" & prior$role==role,];stopifnot(nrow(old)==1)
  mask <- population==group; new <- mean((p[mask]-y[mask])^2)
  comparisons[[length(comparisons)+1]] <- c(group,role,old$n_scored,old$brier,as.character(sum(mask)),sprintf("%.17g",new),sprintf("%.17g",new-as.numeric(old$brier)),"descriptive_different_recruitment_populations")
}
write <- function(name,header,rows) {
  dir.create(out,recursive=TRUE,showWarnings=FALSE); matrix <- do.call(rbind,rows);colnames(matrix)<-header
  write.table(matrix,file.path(out,name),sep=",",row.names=FALSE,col.names=TRUE,quote=FALSE,na="")
}
write("sealed_predictions.csv",c("student_id","college_id","population","frame_probability","comparator_probability"),
      lapply(seq_len(nrow(raw)),function(i)c(raw$student_id[i],raw$college_id[i],population[i],sprintf("%.17g",p[i]),sprintf("%.17g",comparator))))
write("sealed_scores.csv",c("population","college_id","method_id","n","events","nonevents","prevalence","mean_probability","brier","log_loss","auc_pairs2","auc_denominator","auc","auc_reason","clipped_n","mean_no_transfer_risk","mean_gap_to_no_transfer_risk","mae_to_no_transfer_risk","rmse_to_no_transfer_risk","risk_reference_reason"),scores)
write("sealed_bins.csv",c("population","method_id","bin","lower","upper","n","events","mean_probability","observed_fraction","mean_no_transfer_risk","reason"),bins)
write("sealed_comparisons.csv",c("population","prior_role","prior_n","prior_brier","sealed_n","sealed_brier","difference","reason"),comparisons)
cat("SEALED_BASE_R_OK predictions=",nrow(raw)," no_refit=true World_draws=0\n",sep="")
