import csv
import gc
import os
import sys
import time
import importlib.util
import numpy as np
import traceback

from numpy.random.mtrand import uniform, seed

from commons import * 
import threading
import _thread as thread

from functionUtils import RESTRICT_INVOCATIONS


def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush() # Python 3 stderr is likely buffered.
    thread.interrupt_main() # raises KeyboardInterrupt

def exit_after(s):
    '''
    use as decorator to exit process if
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer

@exit_after(60)
def run_func(fnc,args):
    return fnc(*args.values())


def test_roots(f1,f2, res, gt, maxerr): 
    
    err = 0
    
    #count non roots 
    res = [x for x in res]
    print(res)

    for x in res: 
        if (abs(f1(x)-f2(x))>maxerr):
            err+=1
            print("ERR:",x,f1(x),f2(x),abs(f1(x) - f2(x)))

    #count missing roots 
    gt = [x for x in gt]
    
    res = np.array(res)
    for gtr in gt: 
        #find the closest entry in res:
        dist = abs(res - gtr)
        i = np.argmin(dist)
        x = float(res[i])
        for y in np.linspace(min(x,gtr),max(x,gtr),10):
            if abs(f1(y) - f2(y)) > maxerr:
               err+=1
               print("MISS:", gtr,x,abs(f1(x) - f2(x)))
               break
    return err

class Grader():
    def __init__(self, dir_path):
        try:
            sys.path.append(dir_path)
        except Exception as e:
            print(e)
        self.reports=[]
        self.dir_path=dir_path
        self.student_file = self.dir_path.split('/')[-1]
        self.student_name = self.student_file.split('_')[0]
        self.student_number = self.student_file.split('_')[1]


    def grade_assignment(self,to_grade_func,params,assignment_name,err_funcs,expected_results,repeats=1):
        execnum=0

        for p,result,func_error in zip(params,expected_results,err_funcs):
            execnum+=1
            report = {
                    'student_moodle_id':self.student_number,
                    'student_name':self.student_name,
                    'assignemtn':assignment_name,
                    'function':to_grade_func.__name__,
                    'expected_output':result,
                    'execnum':execnum,            
                    }
            report = {**report,**p}
            
            
            error = "None"
            res = "None"
            start = time.time()

            try:
             start = time.time()
             for i in range(repeats):
                 gc.collect()
                 res=run_func(to_grade_func,p)
             end = time.time()
             error=func_error(res,result)
             print(res,result,error)
            #except KeyboardInterrupt:
            #    print('got keyboard interupt')
            #    end = time.time()
            #    error='timeout'
            except (KeyboardInterrupt,Exception) as e:
                end = time.time()
                errors=traceback.format_exc()
                # print(errors)
                error=str(e)


            report['output']=str(res)
            report['error']=error
            report['repeats']=repeats
            report['time']=end-start
            report['mark']="ADDME"
            
            self.reports.append(report)

    def add_error_report(self,assignment,place,error,repeates):
        report = {'student_moodle_id': self.student_number, 'student_name': self.student_name,
                  'assignemtn': assignment, 'function': 'ERROR', 'output': 'ERROR',
                  'expected_output': 'ERROR', 'error': error, 'repeats': repeates, 'time': 'ERROR',
                  'mark': "0"}
        self.reports.append(report)

    def grade_assignment_1(self):
        try:
            import assignment1

            R=RESTRICT_INVOCATIONS
            names=  ('f','a','b','n')
            valss=[(R(10)(f2) ,0  ,5  ,10 ),
                   (R(20)(f4) ,-2 ,4  ,20 ),
                   (R(50)(f3) ,-1 ,5  ,50 ),
                   (R(20)(f13),3  ,10 ,20 ),
                   (R(20)(f1) ,2  ,5  ,20 ),
                   (R(10)(f7) ,3  ,16 ,10 ),
                   (R(10)(f8) ,1  ,3  ,10 ),
                   (R(10)(f9) ,5  ,10 ,10 ),
                   ]
            params=[dict(zip(names,vals)) for vals in valss]
            
            expected_results=[f2,f4,f3,f13,f1,f7,f8,f9]
            
            func_error=[ #mean absolute error at 2n points within the [a,b] range
                    SAVEARGS(a=a,b=b,n=n)(
                        lambda fres,fexp,a,b,n: 
                            sum([
                                    abs(fres(x)-fexp(x)) 
                                    for x in uniform(low=a,high=b,size=2*n)
                                    ])/2/n 
                        )
                    for _,a,b,n in valss
                    ]

            repeats=1
       
            ass = assignment1.Assignment1()
            self.grade_assignment(ass.interpolate,params,'Assignment 1',func_error,expected_results,repeats)
        except Exception as e:
            self.add_error_report('Assignment 1', 'interpolate', e, 1)


    def grade_assignment_2(self):
        try:
            import assignment2


            names= ('f1' ,'f2'  ,'a' ,'b')
            valss=[(f2_nr,f3_nr ,0.5 ,2 ),
                   (f3_nr,f10   ,1   ,10 ),
                   (f1   ,f2_nr ,-2  ,5  ),
                   (f12  ,f13   ,-0.5,1.5)
                   ]
            params=[dict(zip(names,vals)) for vals in valss] 

            expected_results=[
                    [0.671718,1.8147],
                    [1.62899,2.69730,3.725809,3.7914655],
                    [-0.79128,3.79128],
                    [-0.175390,1.42539]
                    ]
            func_error=[ #total number of non roots and missing roots
                    SAVEARGS(f1=f1,f2=f2,a=a,b=b)(
                            lambda res,exp,f1,f2,a,b: 
                                test_roots(f1,f2,res,exp,0.001)
                            )
                    for f1,f2,a,b in valss
                    ]
            repeats=15

            ass = assignment2.Assignment2()
            self.grade_assignment(ass.intersections,params,'Assignment 2',func_error,expected_results,repeats)
        except Exception as e:
            self.add_error_report('Assignment 2', 'intersections', e, 1)

    def grade_assignment_3(self):
        try:
            import assignment3

            names= ('f'   ,'a','b','n')
            valss=[(f1    ,2  ,5  ,4  ),
                   (f2    ,2  ,10 ,10 ),
                   (f3    ,1  ,1.5,4  ),
                   (f13_nr,2  ,4  ,20 ),
                   (f12_nr,5  ,8  ,10 )]
            params=[dict(zip(names,vals)) for vals in valss]
            expected_results=[
                    15,
                    202.6666666666667,
                    0.469565,
                    35.3333333,
                    37.5]
            func_error=[
                    SAVEARGS(f=f,a=a,b=b,n=n)(
                            lambda res,expected,f,a,b,n:
                                abs(res-expected)
                            )
                    for f,a,b,n in valss
                    ]

            repeats=15
            ass = assignment3.Assignment3()
            self.grade_assignment(ass.integrate,params,'Assignment 3 integrate',func_error,expected_results,repeats)
        except Exception as e:
            self.add_error_report('Assignment 3', 'integrate', e, 1)

    def grade_assignment_3_areabetween(self):
        try:
            import assignment3

            repeats=3
            R=RESTRICT_INVOCATIONS
            names= ('f1','f2')
            valss=[
                   (f10 ,f2  ),
                   ]
            params=[dict(zip(names,vals)) for vals in valss]
            expected_results=[
                    0.731257,
                    ]
            func_error=[
                    SAVEARGS(f1=f1,f2=f2)(
                            lambda res,expected,f1,f2:
                                abs(res-expected)
                            )
                    for f1,f2 in valss
                    ]

            
            ass = assignment3.Assignment3()
            self.grade_assignment(ass.areabetween,params,'Assignment 3 areabetween',func_error,expected_results,repeats)
        except Exception as e:
            self.add_error_report('Assignment 3 areabetween', 'areabetween', e, 1)



    def grade_assignment_4(self):
        try:
            import assignment4

            names= ('f'      ,'a','b','d','maxtime')
            valss=[(f2_noise ,0  ,5  ,10 ,5),
                   (f4_noise ,-2 ,4  ,20 ,10),
                   (f3_noise ,-1 ,5  ,50 ,20),
                   (f13_noise,3  ,10 ,20 ,15),
                   (f1_noise ,2  ,5  ,20 ,20),
                   (f7_noise ,3  ,16 ,10 ,10),
                   (f8_noise ,1  ,3  ,10 ,15),
                   (f9_noise ,5  ,10 ,10 ,20)
                   ]
            params=[dict(zip(names,vals)) for vals in valss] 
            expected_results=[f2,f4,f3,f13,f1,f7,f8,f9]
            
            func_error=[ #mean absolute error at 2n points within the [a,b] range
                    SAVEARGS(f=f,a=a,b=b,n=n,t=t)(
                            lambda fres,fexp,f,a,b,n,t: 
                                sum([
                                        abs(fres(x)-fexp(x)) 
                                        for x in uniform(low=a,high=b,size=2*n)
                                        ])/2/n 
                            )
                    for f,a,b,n,t in valss
                    ]
           
            repeats=1

            ass = assignment4.Assignment4()
            self.grade_assignment(ass.fit,params,'Assignment 4',func_error,expected_results,repeats)
        except Exception as e:
            self.add_error_report('Assignment 4', 'fit', e, 1)


    def grade_assignment_5_area(self):
        try:
            import assignment5

            names= ('contour'         ,'maxerr')
            valss=[
                    (shape1().contour ,0.001),
                    (shape2().contour ,0.001),
                    (shape3().contour ,0.001),
                    (shape4().contour ,0.001),
                    (shape5().contour ,0.001),
                    (shape6().contour ,0.001),
                    (shape7().contour ,0.001),
                   ]
            params=[dict(zip(names,vals)) for vals in valss] 
            
            expected_results=[
                    shape1().area(),
                    shape2().area(),
                    shape3().area(),
                    shape4().area(),
                    shape5().area(),
                    shape6().area(),
                    shape7().area(),
                    ]
            
            func_error=[ 
                    lambda res,exp: 
                                abs(abs(res)-abs(exp))/abs(exp) 
                    for c,e in valss
                    ]

            repeats=1
       

            ass = assignment5.Assignment5()
            self.grade_assignment(ass.area,params,'Assignment 5 area',func_error,expected_results,repeats)

        except Exception as e:
            self.add_error_report('Assignment 5 area', 'area', e, 1)
            
    def grade_assignment_5_fit(self):
        try: 
            import assignment5


            names= ('sample'         ,'maxtime')
            valss=[
                    (shape1().sample ,20),
                    (shape2().sample ,20),
                    (shape3().sample ,20),
                    (shape4().sample ,20),
                    (shape5().sample ,20),
                    (shape6().sample ,20),
                    (shape7().sample ,20),
                   ]
            params=[dict(zip(names,vals)) for vals in valss] 
            
            expected_results=[
                    shape1().area(),
                    shape2().area(),
                    shape3().area(),
                    shape4().area(),
                    shape5().area(),
                    shape6().area(),
                    shape7().area()
                    ]
            
            func_error=[ 
                    lambda res,exp: 
                        abs(abs(res.area())-abs(exp))/abs(exp) 
                    for c,e in valss
                    ]
           
            repeats=1
 
            ass = assignment5.Assignment5()
            self.grade_assignment(ass.fit_shape,params,'Assignment 5 fit',func_error,expected_results,repeats)
        except Exception as e:
            self.add_error_report('Assignment 5 fit', 'fit', e, 1)

    def report(self):
        with open(os.path.join(self.dir_path,'res.csv'),'w',newline='', encoding='utf-16') as f:
            fieldnames = ['student_moodle_id','student_name','assignemtn','function','execnum', 'f','f1','f2','contour','sample','a','b','n','d','maxtime','maxerr','output','expected_output','error','repeats','time','mark']

            writer=csv.DictWriter(f,delimiter='\t',fieldnames=fieldnames)
            writer.writeheader()
            # writer.writerow(['student_moodle_id','student_name','assignemtn','function','params','output','expected_output','error','repeats','time','mark'])
            for report in self.reports:
                writer.writerow(report)


    def grade(self):
        self.grade_assignment_1()
        self.grade_assignment_2()
        self.grade_assignment_3()
        self.grade_assignment_4()
        self.grade_assignment_5_area()
        self.grade_assignment_5_fit()
        self.report()
        sys.path.remove(self.dir_path)

if __name__=='__main__':

    grdr=Grader(os.path.dirname(os.path.abspath(__file__)))
    grdr.grade()
    grdr.report()
    
