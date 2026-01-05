%% initially trying neural net structure with logical one and it doesn't work :(
x=[0,0;1,0;0,1;1,1];
ystar=[0;1;1;1];
w=rand(1,2)
r=1;
for i=1:4
    ah=w(1)*x(i,1)+w(2)*x(i,2)
    if ah>=0
        y=1;
    else
        y=0;
    end
    ystar(i)-y
    (ystar(i)-y)*x(i,:)
    w=w+r*(ystar(i)-y)*x(i,:)
end

for i=1:4
    ah=w(1)*x(i,1)+w(2)*x(i,2);
    if ah>=0
        y=1;
    else
        y=0;
    end
y
end
%% load data, initialize test/train trials, initialize expected output
data=load('recitation_data.mat');

percent_test=.8;
test_num = randperm(50,percent_test*50); 
train_trials=[test_num test_num+50 test_num+100 test_num+150 test_num+200 test_num+250 test_num+300 test_num+350];
train_trials=train_trials(randperm(length(train_trials)));
test_trials=linspace(1,400,400);
test_trials=test_trials(~ismember(linspace(1,400,400),train_trials));
test_trials=test_trials(randperm(length(test_trials)));

ystar=zeros(400,2);
ystar(1:50,1)=1;
ystar(1:50,2)=0;
ystar(51:100,1)=1;%0.7071;
ystar(51:100,2)=1;%0.7071;
ystar(101:150,1)=0;
ystar(101:150,2)=1;
ystar(151:200,1)=-1;%-0.7071;
ystar(151:200,2)=1;%0.7071;
ystar(201:250,1)=-1;
ystar(201:250,2)=0;
ystar(251:300,1)=-1;%-0.7071;
ystar(251:300,2)=-1;%-0.7071;
ystar(301:350,1)=0;
ystar(301:350,2)=-1;
ystar(351:400,1)=1;%0.7071;
ystar(351:400,2)=-1;%-0.7071;
%% Movement Planning
onezero=data.movement_direction(1).neuron_activity.movement_planning;
seven=data.movement_direction(2).neuron_activity.movement_planning;
zeroone=data.movement_direction(3).neuron_activity.movement_planning;
onegseven=data.movement_direction(4).neuron_activity.movement_planning;
negonezero=data.movement_direction(5).neuron_activity.movement_planning;
negnegseven=data.movement_direction(6).neuron_activity.movement_planning;
zeronegone=data.movement_direction(7).neuron_activity.movement_planning;
sevenoneg=data.movement_direction(8).neuron_activity.movement_planning;
x_move_plan=[onezero;seven;zeroone;onegseven;negonezero;negnegseven;zeronegone;sevenoneg];


weights=rand(2,10);
alpha=0.5;
for i=1:length(train_trials)
    index=train_trials(i);
    y1=dot(x_move_plan(index,:), weights(1,:));
    y2=dot(x_move_plan(index,:), weights(2,:));
    % less then -0.5 output=-1, if >0.5 output=1, else =0
    if y1<=-0.5
        y1=-1;
    elseif y1>=0.5
        y1=1;
    else 
        y1=0;
    end
     if y2<=-0.5
        y2=-1;
    elseif y2>=0.5
        y2=1;
     else 
        y2=0;
     end
    weights(1,:)=weights(1,:)+alpha*(x_move_plan(index,:).*ystar(index,1)-weights(1,:)*ystar(index,1)^2);
    weights(2,:)=weights(2,:)+alpha*(x_move_plan(index,:).*ystar(index,2)-weights(2,:)*ystar(index,2)^2);
end

total_correct=0;
errors=zeros(1,length(test_trials));
for i=1:length(test_trials)
    index=test_trials(i);
    y1=dot(x_move_plan(index,:), weights(1,:));
    y2=dot(x_move_plan(index,:), weights(2,:));
     if y1<=-0.5
        y1=-1;
    elseif y1>=0.5
        y1=1;
    else 
        y1=0;
    end
     if y2<=-0.5
        y2=-1;
    elseif y2>=0.5
        y2=1;
     else 
        y2=0;
     end
     errors(i)=sqrt((ystar(index,1)-y1)^2+(ystar(index,2)-y2)^2);
     if y1==ystar(index,1) && y2==ystar(index,2)
         total_correct=total_correct+1;
     end
end
percent_correct=total_correct/length(test_trials)*100
avg_error=mean(errors)
%% Movement Execution
onezero=data.movement_direction(1).neuron_activity.movement_execution;
seven=data.movement_direction(2).neuron_activity.movement_execution;
zeroone=data.movement_direction(3).neuron_activity.movement_execution;
onegseven=data.movement_direction(4).neuron_activity.movement_execution;
negonezero=data.movement_direction(5).neuron_activity.movement_execution;
negnegseven=data.movement_direction(6).neuron_activity.movement_execution;
zeronegone=data.movement_direction(7).neuron_activity.movement_execution;
sevenoneg=data.movement_direction(8).neuron_activity.movement_execution;
x_move_exec=[onezero;seven;zeroone;onegseven;negonezero;negnegseven;zeronegone;sevenoneg];

weights=rand(2,10);
alpha=0.12;
for i=1:length(train_trials)
    index=train_trials(i);
    y1=dot(x_move_exec(index,:), weights(1,:));
    y2=dot(x_move_exec(index,:), weights(2,:));
    % less then -0.5 output=-1, if >0.5 output=1, else =0
    if y1<=-0.5
        y1=-1;
    elseif y1>=0.5
        y1=1;
    else 
        y1=0;
    end
     if y2<=-0.5
        y2=-1;
    elseif y2>=0.5
        y2=1;
     else 
        y2=0;
     end
    weights(1,:)=weights(1,:)+alpha*(x_move_exec(index,:).*ystar(index,1)-weights(1,:)*ystar(index,1)^2);
    weights(2,:)=weights(2,:)+alpha*(x_move_exec(index,:).*ystar(index,2)-weights(2,:)*ystar(index,2)^2);
end

total_correct=0;
errors=zeros(1,length(test_trials));
for i=1:length(test_trials)
    index=test_trials(i);
    y1=dot(x_move_exec(index,:), weights(1,:));
    y2=dot(x_move_exec(index,:), weights(2,:));
     if y1<=-0.5
        y1=-1;
    elseif y1>=0.5
        y1=1;
    else 
        y1=0;
    end
     if y2<=-0.5
        y2=-1;
    elseif y2>=0.5
        y2=1;
     else 
        y2=0;
     end
     errors(i)=sqrt((ystar(index,1)-y1)^2+(ystar(index,2)-y2)^2);
     if y1==ystar(index,1) && y2==ystar(index,2)
         total_correct=total_correct+1;
     end
end
percent_correct=total_correct/length(test_trials)*100
avg_error=mean(errors)
%% Static Hold
% static hold neural activity for location (1,0)
onezero=data.movement_direction(1).neuron_activity.static_hold;
seven=data.movement_direction(2).neuron_activity.static_hold;
zeroone=data.movement_direction(3).neuron_activity.static_hold;
onegseven=data.movement_direction(4).neuron_activity.static_hold;
negonezero=data.movement_direction(5).neuron_activity.static_hold;
negnegseven=data.movement_direction(6).neuron_activity.static_hold;
zeronegone=data.movement_direction(7).neuron_activity.static_hold;
sevenoneg=data.movement_direction(8).neuron_activity.static_hold;
x_static_hold=[onezero;seven;zeroone;onegseven;negonezero;negnegseven;zeronegone;sevenoneg];

weights=rand(2,10);
alpha=0.5;
for i=1:length(train_trials)
    index=train_trials(i);
    y1=dot(x_static_hold(index,:), weights(1,:));
    y2=dot(x_static_hold(index,:), weights(2,:));
    % less then -0.5 output=-1, if >0.5 output=1, else =0
    if y1<=-0.5
        y1=-1;
    elseif y1>=0.5
        y1=1;
    else 
        y1=0;
    end
     if y2<=-0.5
        y2=-1;
    elseif y2>=0.5
        y2=1;
     else 
        y2=0;
     end
    weights(1,:)=weights(1,:)+alpha*(x_static_hold(index,:).*ystar(index,1)-weights(1,:)*ystar(index,1)^2);
    weights(2,:)=weights(2,:)+alpha*(x_static_hold(index,:).*ystar(index,2)-weights(2,:)*ystar(index,2)^2);
end

total_correct=0;
errors=zeros(1,length(test_trials));
for i=1:length(test_trials)
    index=test_trials(i);
    y1=dot(x_static_hold(index,:), weights(1,:));
    y2=dot(x_static_hold(index,:), weights(2,:));
     if y1<=-0.5
        y1=-1;
    elseif y1>=0.5
        y1=1;
    else 
        y1=0;
    end
     if y2<=-0.5
        y2=-1;
    elseif y2>=0.5
        y2=1;
     else 
        y2=0;
     end
     errors(i)=sqrt((ystar(index,1)-y1)^2+(ystar(index,2)-y2)^2);
     if y1==ystar(index,1) && y2==ystar(index,2)
         total_correct=total_correct+1;
     end
end
percent_correct=total_correct/length(test_trials)*100
avg_error=mean(errors)
%% unsupervised
onezero=data.movement_direction(1).neuron_activity.movement_execution;
seven=data.movement_direction(2).neuron_activity.movement_execution;
zeroone=data.movement_direction(3).neuron_activity.movement_execution;
onegseven=data.movement_direction(4).neuron_activity.movement_execution;
negonezero=data.movement_direction(5).neuron_activity.movement_execution;
negnegseven=data.movement_direction(6).neuron_activity.movement_execution;
zeronegone=data.movement_direction(7).neuron_activity.movement_execution;
sevenoneg=data.movement_direction(8).neuron_activity.movement_execution;
x_move_exec=[onezero;seven;zeroone;onegseven;negonezero;negnegseven;zeronegone;sevenoneg];

weights=rand(2,10);
alpha=0.5;
for i=1:length(train_trials)
    index=train_trials(i);
    y1=dot(x_move_exec(index,:), weights(1,:));
    y2=dot(x_move_exec(index,:), weights(2,:));
    % less then -0.5 output=-1, if >0.5 output=1, else =0
    if y1<=-0.5
        y1=-1;
    elseif y1>=0.5
        y1=1;
    else 
        y1=0;
    end
     if y2<=-0.5
        y2=-1;
    elseif y2>=0.5
        y2=1;
     else 
        y2=0;
     end
    weights(1,:)=weights(1,:)+alpha*(x_move_exec(index,:).*y1-weights(1,:)*y1^2);
    weights(2,:)=weights(2,:)+alpha*(x_move_exec(index,:).*y2-weights(2,:)*y2^2);
end

total_correct=0;
errors=zeros(1,length(test_trials));
for i=1:length(test_trials)
    index=test_trials(i);
    y1=dot(x_move_exec(index,:), weights(1,:));
    y2=dot(x_move_exec(index,:), weights(2,:));
     if y1<=-0.5
        y1=-1;
    elseif y1>=0.5
        y1=1;
    else 
        y1=0;
    end
     if y2<=-0.5
        y2=-1;
    elseif y2>=0.5
        y2=1;
     else 
        y2=0;
     end
     errors(i)=sqrt((ystar(index,1)-y1)^2+(ystar(index,2)-y2)^2);
     if y1==ystar(index,1) && y2==ystar(index,2)
         total_correct=total_correct+1;
     end
end
percent_correct=total_correct/length(test_trials)*100
avg_error=mean(errors)
%% delta
onezero=data.movement_direction(1).neuron_activity.movement_execution;
seven=data.movement_direction(2).neuron_activity.movement_execution;
zeroone=data.movement_direction(3).neuron_activity.movement_execution;
onegseven=data.movement_direction(4).neuron_activity.movement_execution;
negonezero=data.movement_direction(5).neuron_activity.movement_execution;
negnegseven=data.movement_direction(6).neuron_activity.movement_execution;
zeronegone=data.movement_direction(7).neuron_activity.movement_execution;
sevenoneg=data.movement_direction(8).neuron_activity.movement_execution;
x_move_exec=[onezero;seven;zeroone;onegseven;negonezero;negnegseven;zeronegone;sevenoneg];

weights=rand(2,10);
alpha=0.5;
for i=1:length(train_trials)
    index=train_trials(i);
    y1=dot(x_move_exec(index,:), weights(1,:));
    y2=dot(x_move_exec(index,:), weights(2,:));
    % less then -0.5 output=-1, if >0.5 output=1, else =0
    if y1<=-0.5
        y1=-1;
    elseif y1>=0.5
        y1=1;
    else 
        y1=0;
    end
     if y2<=-0.5
        y2=-1;
    elseif y2>=0.5
        y2=1;
     else 
        y2=0;
     end
    weights(1,:)=weights(1,:)+alpha*(ystar(index,1)-y1).*x_move_exec(index,:);
    weights(2,:)=weights(2,:)+alpha*(ystar(index,2)-y2).*x_move_exec(index,:);
end

total_correct=0;
errors=zeros(1,length(test_trials));
for i=1:length(test_trials)
    index=test_trials(i);
    y1=dot(x_move_exec(index,:), weights(1,:));
    y2=dot(x_move_exec(index,:), weights(2,:));
     if y1<=-0.5
        y1=-1;
    elseif y1>=0.5
        y1=1;
    else 
        y1=0;
    end
     if y2<=-0.5
        y2=-1;
    elseif y2>=0.5
        y2=1;
     else 
        y2=0;
     end
     errors(i)=sqrt((ystar(index,1)-y1)^2+(ystar(index,2)-y2)^2);
     if y1==ystar(index,1) && y2==ystar(index,2)
         total_correct=total_correct+1;
     end
end
percent_correct=total_correct/length(test_trials)*100
avg_error=mean(errors)