#grade
#先做基础功能
import json
import matplotlib.pyplot as plt
import time
from .EED import EED
import multiprocessing
from tabulate import tabulate
progress=0
def write_log(s,file='./logs.txt'):
    with open(file,'a') as f:
        f.write(s+'\n')


processing_lis=[]
processed_list=[]
with open('logging.txt','w',encoding='utf-8') as f:
    f.write('')
def process_single_problem(data):
    model_name=data['model']
    ai_ans=data['model_answer']
    right_ans=data['right_answer']
    problem_id=data['id']

    scoring_pars=data['scoring_pars']

    t0=time.time()


    score,rel_distance,treesize,distance_num=EED(right_ans,ai_ans,debug_mode=False,scoring_parameters=scoring_pars)
    t1=time.time()

    with open('logging.txt','a',encoding='utf-8') as f:
        f.write(f"Finished processed{model_name}Problem{data['id']},Time{t1-t0}\n")

    return [model_name,score,problem_id,rel_distance,treesize,distance_num]
def main(gt_file_dir,gen_file_dir,output_dir,parameters):
    """
    final_answer_f="./solutions/dsr1.json"
    approved_problems_f="god_answer.json"
    output_path="./output_data.json"
    """
    if not parameters:
        parameters=[60,100]

    final_answer_f=gt_file_dir
    approved_problems_f=gen_file_dir
    output_path=output_dir

    with open(final_answer_f, "r", encoding='utf-8') as f:
        final_answer = json.load(f)
    
    with open(approved_problems_f, "r", encoding='utf-8') as f:
        approved_problems = json.load(f)

    approved_problems_dict={}
    for data in approved_problems:
        data['model_name']=[]
        data['model_score']=[]
        data['model_answer']=[]
        data['model_distance']=[]
        data['model_score_var']=0
        data['answer_size']=0
        approved_problems_dict[data['id']]=data

    model_list=[]
    final_answer_dict={}
    for data in final_answer:
        name=data['model']
        if not name in model_list:
            model_list.append(name)

        final_answer_dict[(data['id'],data['model'])]=data


    work_list=[]
    

    for answers in approved_problems[0:]:
        if answers['id']==108:
            continue
        for model in model_list:

            id_number=answers['id']
            #print(id_number,model)
            query_answer=(id_number,model)
            if query_answer in final_answer_dict:
                model_answer=final_answer_dict[(id_number,model)]['answer']
                right_answer=approved_problems_dict[id_number]['answer']

                work_list.append({'id':id_number,'model':model,'model_answer':model_answer,'right_answer':right_answer,'scoring_pars':parameters})
    
    
    print(f"Successfully Built Worklist,total_length:{len(work_list)}")


    #scoring
    #根据final_answer中的模型回答和approved_problems中的答案进行评分,修改final_answer文件
    
    cpu_cores = multiprocessing.cpu_count()
    print(f"We have {cpu_cores} cores for grading...")
    t0=time.time()
    #cpu_cores=1
    results=[]

    
    with multiprocessing.Pool(processes=cpu_cores) as pool:
        # 提交任务并获取结果（按顺序）
        results = list(pool.map(process_single_problem, work_list))
    
        
    t1=time.time()
    print(f"Grading Finished,total time:{t1-t0}")

    #plot
    model_scores,model_nums={},{}
    for name in model_list:
        model_scores[name]=0
        model_nums[name]=0
    num=len(approved_problems)


    dist_data=[]
    for result in results:
        model=result[0]
        score_i=result[1]
        problem_id=result[2]
        rel_dist=result[3]
        tree_size=result[4]
        distance_number=result[5]

        model_scores[model]+=score_i
        model_nums[model]+=1

        approved_problems_dict[problem_id]['answer_size']=max(tree_size,approved_problems_dict[problem_id]['answer_size'])
        approved_problems_dict[problem_id]['model_distance'].append(distance_number)
        approved_problems_dict[problem_id]['model_name'].append(model)
        approved_problems_dict[problem_id]['model_score'].append(score_i)
        approved_problems_dict[problem_id]['model_answer'].append(final_answer_dict[(problem_id,model)]['answer'])

        dist_data.append(rel_dist)
        #print(model,result[1],problem_id)

    for data in approved_problems:

        score_list=data['model_score']
        if score_list:
            avg_score=sum(score_list)/len(score_list)
            score_2=0
            for score in score_list:
                score_2+=(score-avg_score)**2
            data['model_score_var']= score_2/len(score_list)
        #print(data)
    
    #存储data为json
    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(approved_problems,f,ensure_ascii=False,indent=4)


    output_table=[]
    for model in model_scores:
        model_scores[model]=model_scores[model]/model_nums[model]
        output_table.append([model,model_scores[model]])
    

    s_opt=tabulate(output_table, headers=['Model','Score'], tablefmt="fancy_grid",floatfmt='.2f')
    print(s_opt)
    #print(model_scores)

    print("Complete!")
        
    return s_opt

if __name__ == "__main__":
    """
    final_answer_f="./solutions/qwen_solution.json"
    approved_problems_f="god_answer.json"
    output_path="./output_data.json"
    """
    main("./solutions/claude-sonnet-4-0514.json","./god_answer.json",f"./data_0531.json",[60,100])
