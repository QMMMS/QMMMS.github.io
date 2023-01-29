drop table if exists t_student;

drop table if exists t_class;

create table t_class (
	classno int primary key,
  classname varchar(255)
);

create table t_student(
	id int primary key auto_increment,
  name varchar(255),
  classno int,
  foreign key(classno) references t_class(classno)
);

insert into t_class values (100, 'class one'), (101, 'class two');

insert into t_student(name, classno) 
values ('jk', 100), ('jjkk', 101), ('ab', 100), ('cd', 101), ('ps', 100), ('ed', 101);

select s.*, t.classname from t_student s join t_class t where s.classno = t.classno;