function OutputVideo(jj,~)
%%Description
%寫成一個function,方便我調畫圖參數，輸出模擬結果
%用全局變數使得在function內可以溝通
%OutputVideo(): intialize video object
%OutputVideo(jj): jj for index of frames
%OutputVideo(jj,~): close video object
global myVideo F
switch nargin
    case 0
    %%initialize video%%
        fps = 8;
        loops = 5000;
        % Preallocate frame structre array
        F = struct('cdata',[],'colormap',[]);
        F(loops) = struct('cdata',[],'colormap',[]);  
        myVideo = VideoWriter('findGS.avi'); %open video file
        myVideo.FrameRate = fps;  %can adjust this, 5 - 10 works well for me
        open(myVideo)
    case 1
    %%Animation part%%
%        %指定要抓frame的範圍
%         ax = gca;
%         ax.Units = 'pixels';
%         pos = ax.Position;
%         ti = ax.TightInset;
%         rect = [-ti(1), -ti(2), pos(3)+ti(1)+ti(3), pos(4)+ti(2)+ti(4)];
        try
            F(jj) = getframe(gcf);
            writeVideo(myVideo, F(jj));
        catch ME
            fprintf('miss%d\n',jj)
            throw(ME)
        end
    case 2
        close(myVideo)
end