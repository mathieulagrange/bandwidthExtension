
genres= {'disco', 'rock', 'classical', 'reggae', 'jazz', 'blues', 'hiphop', 'metal', 'pop', 'country'};

path = '/data/databases/music/gtzan/';

system (['rm ' path 'test/*']);
system (['rm ' path 'dev/*']);

for k=1:length(genres)
    for l=0:99
    if (l<10)
        add0 = '0';
    else
        add0 ='';
    end
    if (l<70)
        dest = 'dev/';
    else
        dest ='test/';
    end
    command = ['ln -s ' path 'genres/' genres{k} '/' genres{k} '.000' add0 num2str(l) '.au ' path dest genres{k} '.000' add0 num2str(l) '.au'];
     system(command);
    end
end